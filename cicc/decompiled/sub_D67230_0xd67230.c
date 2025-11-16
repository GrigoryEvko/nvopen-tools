// Function: sub_D67230
// Address: 0xd67230
//
__m128i *__fastcall sub_D67230(__m128i *a1, unsigned __int8 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  int v8; // edx
  unsigned int v9; // r12d
  int v10; // eax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  int v15; // ebx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // edx
  int v20; // edx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  _DWORD *v35; // rax
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  __m128i v38; // xmm2
  unsigned int v39; // [rsp+0h] [rbp-80h]
  char v40; // [rsp+7h] [rbp-79h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  int v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  int v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  __m128i v47; // [rsp+20h] [rbp-60h] BYREF
  __m128i v48; // [rsp+30h] [rbp-50h] BYREF
  __m128i v49[4]; // [rsp+40h] [rbp-40h] BYREF

  if ( !sub_B49E80((__int64)a2) )
    goto LABEL_2;
  if ( (a2[7] & 0x80u) != 0 )
  {
    v4 = sub_BD2BC0((__int64)a2);
    v6 = v4 + v5;
    v7 = 0;
    if ( (a2[7] & 0x80u) != 0 )
      v7 = sub_BD2BC0((__int64)a2);
    if ( (unsigned int)((v6 - v7) >> 4) )
      goto LABEL_2;
  }
  v8 = *a2;
  v40 = 0;
  v9 = 0;
  v39 = 0;
  v46 = 0;
  v10 = v8 - 29;
  if ( v8 == 40 )
    goto LABEL_21;
LABEL_9:
  v11 = 0;
  if ( v10 != 56 )
  {
    if ( v10 != 5 )
LABEL_70:
      BUG();
    v11 = 64;
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
LABEL_13:
    v12 = sub_BD2BC0((__int64)a2);
    v14 = v12 + v13;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v14 >> 4) )
        goto LABEL_22;
    }
    else
    {
      if ( !(unsigned int)((v14 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_22;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v15 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v16 = sub_BD2BC0((__int64)a2);
        v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
        goto LABEL_18;
      }
    }
    BUG();
  }
LABEL_22:
  while ( 1 )
  {
    v18 = 0;
LABEL_18:
    if ( v9 >= (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v11 - v18) >> 5) )
      break;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a2[32 * (v9 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))] + 8LL)
                  + 8LL) != 14 )
      goto LABEL_20;
    v20 = *a2;
    if ( v20 == 40 )
    {
      v21 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v21 = 0;
      if ( v20 != 85 )
      {
        if ( v20 != 34 )
          goto LABEL_70;
        v21 = 64;
      }
    }
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_53;
    v22 = sub_BD2BC0((__int64)a2);
    v42 = v23 + v22;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v42 >> 4) )
LABEL_73:
        BUG();
LABEL_53:
      v26 = 0;
      goto LABEL_33;
    }
    if ( !(unsigned int)((v42 - sub_BD2BC0((__int64)a2)) >> 4) )
      goto LABEL_53;
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_73;
    v43 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
    if ( (a2[7] & 0x80u) == 0 )
      BUG();
    v24 = sub_BD2BC0((__int64)a2);
    v26 = 32LL * (unsigned int)(*(_DWORD *)(v24 + v25 - 4) - v43);
LABEL_33:
    if ( v9 < (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v21 - v26) >> 5)
      && (unsigned __int8)sub_B49B80((__int64)a2, v9, 81) )
    {
      goto LABEL_20;
    }
    v27 = *a2;
    if ( v27 == 40 )
    {
      v28 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v28 = 0;
      if ( v27 != 85 )
      {
        if ( v27 != 34 )
          goto LABEL_70;
        v28 = 64;
      }
    }
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_57;
    v29 = sub_BD2BC0((__int64)a2);
    v44 = v30 + v29;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v44 >> 4) )
LABEL_74:
        BUG();
LABEL_57:
      v33 = 0;
      goto LABEL_45;
    }
    if ( !(unsigned int)((v44 - sub_BD2BC0((__int64)a2)) >> 4) )
      goto LABEL_57;
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_74;
    v45 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
    if ( (a2[7] & 0x80u) == 0 )
      BUG();
    v31 = sub_BD2BC0((__int64)a2);
    v33 = 32LL * (unsigned int)(*(_DWORD *)(v31 + v32 - 4) - v45);
LABEL_45:
    if ( v9 >= (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v28 - v33) >> 5) )
    {
      v35 = (_DWORD *)sub_B49810((__int64)a2, v9);
      if ( *(_DWORD *)(*(_QWORD *)v35 + 8LL)
        || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a2[32 * (v9 - v35[2])
                                               + 32
                                               * ((unsigned int)v35[2]
                                                - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))]
                                + 8LL)
                    + 8LL) != 14 )
      {
LABEL_47:
        if ( !sub_CF49B0(a2, v9, 50) )
        {
          if ( v46 )
          {
            v40 = 0;
            v34 = *(_QWORD *)&a2[32 * (v9 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
            if ( v46 != v34 || !v34 )
              goto LABEL_2;
          }
          else
          {
            v39 = v9;
            v40 = 1;
            v46 = *(_QWORD *)&a2[32 * (v9 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
          }
        }
      }
    }
    else if ( !(unsigned __int8)sub_B49B80((__int64)a2, v9, 51) )
    {
      goto LABEL_47;
    }
LABEL_20:
    v19 = *a2;
    ++v9;
    v10 = v19 - 29;
    if ( v19 != 40 )
      goto LABEL_9;
LABEL_21:
    v11 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    if ( (a2[7] & 0x80u) != 0 )
      goto LABEL_13;
  }
  if ( !v46 )
  {
LABEL_2:
    a1[3].m128i_i8[0] = 0;
    return a1;
  }
  if ( v40 )
  {
    sub_D669C0(&v47, (__int64)a2, v39, a3);
    v36 = _mm_loadu_si128(&v47);
    v37 = _mm_loadu_si128(&v48);
    a1[3].m128i_i8[0] = 1;
    v38 = _mm_loadu_si128(v49);
    *a1 = v36;
    a1[1] = v37;
    a1[2] = v38;
  }
  else
  {
    sub_B91FC0(v47.m128i_i64, (__int64)a2);
    a1[3].m128i_i8[0] = 1;
    a1->m128i_i64[1] = -1;
    a1->m128i_i64[0] = v46;
    a1[1] = v47;
    a1[2] = v48;
  }
  return a1;
}
