// Function: sub_2C115C0
// Address: 0x2c115c0
//
void __fastcall sub_2C115C0(__int64 a1)
{
  __int64 *v2; // r15
  unsigned __int64 v3; // r12
  __int64 v4; // r14
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // r8d
  _QWORD *v9; // [rsp+8h] [rbp-48h]
  __int64 v10[7]; // [rsp+18h] [rbp-38h] BYREF

  *(_QWORD *)(a1 - 40) = &unk_4A24778;
  *(_QWORD *)a1 = &unk_4A247B8;
  *(_QWORD *)(a1 + 72) = &unk_4A247F0;
  v9 = (_QWORD *)(a1 - 40);
  sub_2BF1E70((_QWORD *)(a1 + 72));
  *(_QWORD *)a1 = &unk_4A23AA8;
  *(_QWORD *)(a1 - 40) = &unk_4A23A70;
  sub_9C6650((_QWORD *)(a1 + 48));
  v2 = *(__int64 **)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23170;
  v3 = (unsigned __int64)&v2[*(unsigned int *)(a1 + 16)];
  if ( v2 != (__int64 *)v3 )
  {
    do
    {
      v4 = *v2;
      v10[0] = a1;
      v5 = *(_QWORD **)(v4 + 16);
      v6 = (__int64)&v5[*(unsigned int *)(v4 + 24)];
      v7 = sub_2C0D780(v5, v6, v10);
      if ( (_QWORD *)v6 != v7 )
      {
        if ( (_QWORD *)v6 != v7 + 1 )
        {
          memmove(v7, v7 + 1, v6 - (_QWORD)(v7 + 1));
          v8 = *(_DWORD *)(v4 + 24);
        }
        *(_DWORD *)(v4 + 24) = v8 - 1;
      }
      ++v2;
    }
    while ( (__int64 *)v3 != v2 );
    v3 = *(_QWORD *)(a1 + 8);
  }
  if ( v3 != a1 + 24 )
    _libc_free(v3);
  sub_2AA7960(v9);
}
