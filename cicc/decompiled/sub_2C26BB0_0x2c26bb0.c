// Function: sub_2C26BB0
// Address: 0x2c26bb0
//
void __fastcall sub_2C26BB0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v3; // rdi
  __int64 *v4; // r15
  unsigned __int64 v5; // r12
  __int64 v6; // r14
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  _QWORD *v9; // rax
  int v10; // r8d
  _QWORD *v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = (_QWORD *)(a1 - 40);
  v3 = (_QWORD *)(a1 + 56);
  v11 = v1;
  *(v3 - 12) = &unk_4A231C8;
  *(v3 - 7) = &unk_4A23200;
  *v3 = &unk_4A23238;
  sub_2BF1E70(v3);
  *(_QWORD *)a1 = &unk_4A23AA8;
  *(_QWORD *)(a1 - 40) = &unk_4A23A70;
  sub_9C6650((_QWORD *)(a1 + 48));
  v4 = *(__int64 **)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23170;
  v5 = (unsigned __int64)&v4[*(unsigned int *)(a1 + 16)];
  if ( v4 != (__int64 *)v5 )
  {
    do
    {
      v6 = *v4;
      v12[0] = a1;
      v7 = *(_QWORD **)(v6 + 16);
      v8 = (__int64)&v7[*(unsigned int *)(v6 + 24)];
      v9 = sub_2C25810(v7, v8, v12);
      if ( (_QWORD *)v8 != v9 )
      {
        if ( (_QWORD *)v8 != v9 + 1 )
        {
          memmove(v9, v9 + 1, v8 - (_QWORD)(v9 + 1));
          v10 = *(_DWORD *)(v6 + 24);
        }
        *(_DWORD *)(v6 + 24) = v10 - 1;
      }
      ++v4;
    }
    while ( (__int64 *)v5 != v4 );
    v5 = *(_QWORD *)(a1 + 8);
  }
  if ( v5 != a1 + 24 )
    _libc_free(v5);
  sub_2AA7960(v11);
  j_j___libc_free_0((unsigned __int64)v11);
}
