// Function: sub_255B060
// Address: 0x255b060
//
void __fastcall sub_255B060(__int64 a1, __int64 a2)
{
  __m128i *v2; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  int v8; // edx
  int v9; // ecx
  int v10; // r8d
  unsigned int v11; // edx
  __int64 v12; // rdi
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // r13
  __int64 v16; // r15
  unsigned int v17; // eax
  unsigned int v18; // esi
  unsigned __int64 v19; // rdi
  __int64 *v20; // rax
  bool v21; // [rsp+1Fh] [rbp-61h]
  __int64 v22; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v23; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h]
  _BYTE v25[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = (__m128i *)(a1 + 72);
  *(_DWORD *)(a1 + 100) = *(_DWORD *)(a1 + 96) | *(_DWORD *)(a1 + 100) & 0x1FF;
  v5 = sub_25096F0((_QWORD *)(a1 + 72));
  v21 = 1;
  if ( !v5 )
    goto LABEL_6;
  v6 = *(_QWORD *)(a2 + 200);
  if ( !*(_DWORD *)(v6 + 40) )
    goto LABEL_5;
  v7 = *(_QWORD *)(v6 + 8);
  v8 = *(_DWORD *)(v6 + 24);
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = 1;
    v11 = (v8 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v12 = *(_QWORD *)(v7 + 8LL * v11);
    if ( v5 != v12 )
    {
      while ( v12 != -4096 )
      {
        v11 = v9 & (v10 + v11);
        v12 = *(_QWORD *)(v7 + 8LL * v11);
        if ( v5 == v12 )
          goto LABEL_5;
        ++v10;
      }
      v21 = 1;
      goto LABEL_6;
    }
LABEL_5:
    v21 = (*(_BYTE *)(v5 + 32) & 0xFu) - 7 > 1;
  }
LABEL_6:
  v23 = (__int64 *)v25;
  v24 = 0x200000000LL;
  LODWORD(v22) = 92;
  sub_2515D00(a2, v2, (int *)&v22, 1, (__int64)&v23, 0);
  v13 = &v23[(unsigned int)v24];
  if ( v23 == v13 )
    goto LABEL_15;
  v14 = a2;
  v15 = v23;
  v16 = v14;
  do
  {
    while ( 1 )
    {
      v17 = sub_A71E40(v15);
      if ( !v17 )
        break;
      if ( (v17 & 0xFFFFFFF3) != 0 )
      {
        if ( (v17 & 0xFFFFFFFC) != 0 )
        {
          if ( (v17 & 0xFFFFFFF0) == 0 )
          {
            if ( !v21 )
            {
              sub_255AA10(&v22, (v17 >> 2) | v17 & 3);
              v18 = v22;
              v19 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
              if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
LABEL_26:
                v19 = *(_QWORD *)(v19 + 24);
LABEL_24:
              v20 = (__int64 *)sub_BD5C60(v19);
              v22 = sub_A77AB0(v20, v18);
              sub_2516380(v16, v2->m128i_i64, (__int64)&v22, 1, 1);
              goto LABEL_11;
            }
            *(_QWORD *)(a1 + 96) |= 0xCC000000CCuLL;
          }
        }
        else
        {
          if ( !v21 )
          {
            v18 = (v17 << 6) | (16 * v17) | v17 | (4 * v17);
            v19 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
            if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
              goto LABEL_26;
            goto LABEL_24;
          }
          *(_QWORD *)(a1 + 96) |= 0xEC000000ECuLL;
        }
      }
      else
      {
        *(_QWORD *)(a1 + 96) |= 0xDC000000DCuLL;
      }
LABEL_11:
      if ( v13 == ++v15 )
        goto LABEL_14;
    }
    ++v15;
    *(_QWORD *)(a1 + 96) |= 0x300000003uLL;
  }
  while ( v13 != v15 );
LABEL_14:
  v13 = v23;
LABEL_15:
  if ( v13 != (__int64 *)v25 )
    _libc_free((unsigned __int64)v13);
}
