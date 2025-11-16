// Function: sub_30F9B30
// Address: 0x30f9b30
//
__int64 __fastcall sub_30F9B30(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // r15
  unsigned __int8 *v5; // r14
  unsigned __int8 v6; // cl
  __int64 v7; // rax
  __int64 v8; // rsi
  int v9; // eax
  int v10; // eax
  unsigned int v11; // edi
  unsigned __int8 **v12; // rdx
  unsigned __int8 *v13; // r8
  unsigned __int8 *v14; // rdx
  unsigned int v15; // edi
  _QWORD *v16; // rdx
  unsigned __int8 *v17; // r8
  __int64 v18; // rax
  __int16 v19; // di
  __int64 v20; // r14
  __int64 v21; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // edi
  unsigned int v26; // ecx
  unsigned __int8 **v27; // rdx
  unsigned __int8 *v28; // r8
  unsigned __int8 **v29; // rax
  unsigned int v30; // r8d
  unsigned __int8 **v31; // rcx
  unsigned __int8 *v32; // r9
  char v33; // al
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  int v37; // eax
  int v38; // edx
  int v39; // r9d
  int v40; // edx
  int v41; // r9d
  int v42; // edx
  int v43; // r9d
  int v44; // ecx
  int v45; // r10d
  __m128i v46; // [rsp+0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+10h] [rbp-70h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  __int64 v49; // [rsp+20h] [rbp-60h]
  __int64 v50; // [rsp+28h] [rbp-58h]
  __int64 v51; // [rsp+30h] [rbp-50h]
  __int64 v52; // [rsp+38h] [rbp-48h]
  __int16 v53; // [rsp+40h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a2 - 64);
  v5 = *(unsigned __int8 **)(a2 - 32);
  v6 = *v4;
  if ( *v4 <= 0x15u )
  {
    if ( *v5 <= 0x15u )
      goto LABEL_11;
    v36 = *(_QWORD *)(a1 + 40);
    v8 = *(_QWORD *)(v36 + 8);
    v37 = *(_DWORD *)(v36 + 24);
    if ( !v37 )
      goto LABEL_11;
    v10 = v37 - 1;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 40);
    v8 = *(_QWORD *)(v7 + 8);
    v9 = *(_DWORD *)(v7 + 24);
    if ( !v9 )
      goto LABEL_10;
    v10 = v9 - 1;
    v11 = v10 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v12 = (unsigned __int8 **)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v4 == *v12 )
    {
LABEL_4:
      v14 = v12[1];
      if ( v14 )
      {
        v6 = *v14;
        v4 = v14;
      }
    }
    else
    {
      v38 = 1;
      while ( v13 != (unsigned __int8 *)-4096LL )
      {
        v39 = v38 + 1;
        v11 = v10 & (v38 + v11);
        v12 = (unsigned __int8 **)(v8 + 16LL * v11);
        v13 = *v12;
        if ( v4 == *v12 )
          goto LABEL_4;
        v38 = v39;
      }
    }
    if ( *v5 <= 0x15u )
      goto LABEL_11;
  }
  v15 = v10 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v16 = (_QWORD *)(v8 + 16LL * v15);
  v17 = (_BYTE *)*v16;
  if ( v5 != (unsigned __int8 *)*v16 )
  {
    v40 = 1;
    while ( v17 != (_BYTE *)-4096LL )
    {
      v41 = v40 + 1;
      v15 = v10 & (v40 + v15);
      v16 = (_QWORD *)(v8 + 16LL * v15);
      v17 = (_BYTE *)*v16;
      if ( v5 == (unsigned __int8 *)*v16 )
        goto LABEL_8;
      v40 = v41;
    }
LABEL_14:
    if ( v6 <= 0x15u )
      goto LABEL_11;
    goto LABEL_15;
  }
LABEL_8:
  if ( !v16[1] )
    goto LABEL_14;
  v5 = (unsigned __int8 *)v16[1];
  if ( v6 <= 0x15u )
    goto LABEL_11;
LABEL_10:
  if ( *v5 <= 0x15u )
    goto LABEL_11;
LABEL_15:
  if ( !sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F) )
  {
    v23 = *(unsigned int *)(a1 + 24);
    v24 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v23 )
    {
      v25 = v23 - 1;
      v26 = (v23 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v27 = (unsigned __int8 **)(v24 + 32LL * v26);
      v28 = *v27;
      if ( *v27 == v4 )
      {
LABEL_18:
        v29 = (unsigned __int8 **)(v24 + 32 * v23);
        if ( v29 != v27 )
        {
          v30 = v25 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v31 = (unsigned __int8 **)(v24 + 32LL * v30);
          v32 = *v31;
          if ( v5 == *v31 )
          {
LABEL_20:
            if ( v29 != v31 && v27[1] == v31[1] )
            {
              v33 = sub_B532C0((__int64)(v27 + 2), v31 + 2, *(_WORD *)(a2 + 2) & 0x3F);
              v34 = sub_AD64A0(*(_QWORD *)(a2 + 8), v33);
              v35 = *(_QWORD *)(a1 + 40);
              v46.m128i_i64[0] = a2;
              *sub_FAA780(v35, v46.m128i_i64) = v34;
              return 1;
            }
          }
          else
          {
            v44 = 1;
            while ( v32 != (unsigned __int8 *)-4096LL )
            {
              v45 = v44 + 1;
              v30 = v25 & (v44 + v30);
              v31 = (unsigned __int8 **)(v24 + 32LL * v30);
              v32 = *v31;
              if ( *v31 == v5 )
                goto LABEL_20;
              v44 = v45;
            }
          }
        }
      }
      else
      {
        v42 = 1;
        while ( v28 != (unsigned __int8 *)-4096LL )
        {
          v43 = v42 + 1;
          v26 = v25 & (v42 + v26);
          v27 = (unsigned __int8 **)(v24 + 32LL * v26);
          v28 = *v27;
          if ( v4 == *v27 )
            goto LABEL_18;
          v42 = v43;
        }
      }
    }
  }
LABEL_11:
  v18 = sub_B43CC0(a2);
  v19 = *(_WORD *)(a2 + 2);
  v46 = (__m128i)(unsigned __int64)v18;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 257;
  v20 = sub_10197D0(v19 & 0x3F, v4, v5, &v46);
  if ( v20 )
  {
    v21 = *(_QWORD *)(a1 + 40);
    v46.m128i_i64[0] = a2;
    *sub_FAA780(v21, v46.m128i_i64) = v20;
    return 1;
  }
  return sub_30F9620(a1, a2);
}
