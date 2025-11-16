// Function: sub_2C26960
// Address: 0x2c26960
//
void __fastcall sub_2C26960(unsigned __int64 a1)
{
  _QWORD *v2; // rdi
  __int64 *v3; // r15
  unsigned __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rax
  int v9; // r8d
  unsigned __int64 v10[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (_QWORD *)(a1 + 96);
  *(v2 - 12) = &unk_4A231C8;
  *(v2 - 7) = &unk_4A23200;
  *v2 = &unk_4A23238;
  sub_2BF1E70(v2);
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  *(_QWORD *)a1 = &unk_4A23A70;
  sub_9C6650((_QWORD *)(a1 + 88));
  v3 = *(__int64 **)(a1 + 48);
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  v4 = (unsigned __int64)&v3[*(unsigned int *)(a1 + 56)];
  if ( v3 != (__int64 *)v4 )
  {
    do
    {
      v5 = *v3;
      v10[0] = a1 + 40;
      v6 = *(_QWORD **)(v5 + 16);
      v7 = (__int64)&v6[*(unsigned int *)(v5 + 24)];
      v8 = sub_2C25810(v6, v7, (__int64 *)v10);
      if ( (_QWORD *)v7 != v8 )
      {
        if ( (_QWORD *)v7 != v8 + 1 )
        {
          memmove(v8, v8 + 1, v7 - (_QWORD)(v8 + 1));
          v9 = *(_DWORD *)(v5 + 24);
        }
        *(_DWORD *)(v5 + 24) = v9 - 1;
      }
      ++v3;
    }
    while ( (__int64 *)v4 != v3 );
    v4 = *(_QWORD *)(a1 + 48);
  }
  if ( v4 != a1 + 64 )
    _libc_free(v4);
  sub_2AA7960((_QWORD *)a1);
  j_j___libc_free_0(a1);
}
