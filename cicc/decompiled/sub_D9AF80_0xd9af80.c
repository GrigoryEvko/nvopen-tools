// Function: sub_D9AF80
// Address: 0xd9af80
//
char __fastcall sub_D9AF80(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 *v6; // rsi
  __int64 v7; // rdx
  _QWORD *i; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // r13
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v14[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+20h] [rbp-30h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 56) = 128;
  v5 = (_QWORD *)sub_C7D670(6144, 8);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = v5;
  v6 = (__int64 *)&unk_49DDFA0;
  v7 = *(unsigned int *)(a1 + 56);
  v14[0] = 0;
  v14[1] = -4096;
  v15 = 0;
  for ( i = &v5[6 * v7]; i != v5; v5 += 6 )
  {
    if ( v5 )
    {
      v5[2] = 0;
      v5[3] = -4096;
      *v5 = &unk_49DDFA0;
      v5[1] = 2;
      v5[4] = v15;
    }
  }
  *(_QWORD *)(a1 + 120) = a3;
  *(_BYTE *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = a2;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 164) = 0;
  v9 = (_QWORD *)sub_22077B0(184);
  v10 = v9;
  if ( v9 )
  {
    v6 = v14;
    LOBYTE(v9) = sub_D9AF00(v9, v14, 0, a2);
  }
  v11 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 128) = v10;
  if ( v11 )
  {
    v12 = *(_QWORD *)(v11 + 40);
    if ( v12 != v11 + 56 )
      _libc_free(v12, v6);
    LOBYTE(v9) = j_j___libc_free_0(v11, 184);
  }
  return (char)v9;
}
