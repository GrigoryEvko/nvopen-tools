// Function: sub_2C11260
// Address: 0x2c11260
//
void __fastcall sub_2C11260(_QWORD *a1)
{
  __int64 *v2; // rcx
  unsigned __int64 v3; // r12
  __int64 *v4; // r15
  __int64 v5; // r13
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rax
  int v9; // r8d
  _QWORD *v10; // [rsp+0h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  *(a1 - 14) = &unk_4A24810;
  *(a1 - 9) = &unk_4A24850;
  *a1 = &unk_4A24888;
  v10 = a1 - 14;
  sub_2BF1E70(a1);
  *(a1 - 9) = &unk_4A23AA8;
  *(a1 - 14) = &unk_4A23A70;
  sub_9C6650(a1 - 3);
  v2 = (__int64 *)*(a1 - 8);
  v11 = (__int64)(a1 - 9);
  *(a1 - 9) = &unk_4A23170;
  v3 = (unsigned __int64)&v2[*((unsigned int *)a1 - 14)];
  if ( v2 != (__int64 *)v3 )
  {
    v4 = v2;
    do
    {
      v5 = *v4;
      v12[0] = v11;
      v6 = *(_QWORD **)(v5 + 16);
      v7 = (__int64)&v6[*(unsigned int *)(v5 + 24)];
      v8 = sub_2C0D780(v6, v7, v12);
      if ( (_QWORD *)v7 != v8 )
      {
        if ( (_QWORD *)v7 != v8 + 1 )
        {
          memmove(v8, v8 + 1, v7 - (_QWORD)(v8 + 1));
          v9 = *(_DWORD *)(v5 + 24);
        }
        *(_DWORD *)(v5 + 24) = v9 - 1;
      }
      ++v4;
    }
    while ( (__int64 *)v3 != v4 );
    v3 = *(a1 - 8);
  }
  if ( (_QWORD *)v3 != a1 - 6 )
    _libc_free(v3);
  sub_2AA7960(v10);
}
