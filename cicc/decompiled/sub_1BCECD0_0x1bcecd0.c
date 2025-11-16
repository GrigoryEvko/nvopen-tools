// Function: sub_1BCECD0
// Address: 0x1bcecd0
//
__int64 __fastcall sub_1BCECD0(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 *v6; // r8
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r9
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r11
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 *v17; // r9
  __int64 *v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 *v23; // r13
  __int64 *v24; // rdi
  __int64 v25; // r9
  __int64 *v26; // r10
  __int64 *v27; // rdx
  __int64 *v28; // rsi
  __int64 v29; // r15
  __int64 v30; // r13
  __int64 v31; // r8
  __int64 *v32; // rsi
  __int64 *v33; // rdi
  __int64 v34; // rdi
  int v35; // esi
  int v36; // eax
  int v37; // edi
  __int64 v38; // r8
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rsi
  unsigned __int8 *v43; // rsi
  __int64 result; // rax
  __int64 v45; // rsi
  __int64 v46; // rsi
  unsigned __int8 *v47; // rsi
  int v48; // edx
  int v49; // r13d
  __int64 v50; // [rsp+0h] [rbp-100h]
  __int64 *v51; // [rsp+8h] [rbp-F8h]
  __int64 v52; // [rsp+8h] [rbp-F8h]
  __int64 v53; // [rsp+18h] [rbp-E8h] BYREF
  unsigned __int8 *v54; // [rsp+20h] [rbp-E0h] BYREF
  __int64 *v55; // [rsp+28h] [rbp-D8h]
  __int64 *v56; // [rsp+30h] [rbp-D0h]
  __int64 v57; // [rsp+38h] [rbp-C8h]
  int v58; // [rsp+40h] [rbp-C0h]
  _BYTE v59[184]; // [rsp+48h] [rbp-B8h] BYREF

  v6 = a2;
  v9 = *(_QWORD *)(*a4 + 40);
  v50 = *a4;
  v10 = *(unsigned int *)(a1 + 1216);
  v53 = v9;
  if ( !(_DWORD)v10 )
    goto LABEL_76;
  v11 = *(_QWORD *)(a1 + 1200);
  v12 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v13 = (__int64 *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( v9 != *v13 )
  {
    v48 = 1;
    while ( v14 != -8 )
    {
      v49 = v48 + 1;
      v12 = (v10 - 1) & (v48 + v12);
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v9 == *v13 )
        goto LABEL_3;
      v48 = v49;
    }
    goto LABEL_76;
  }
LABEL_3:
  v15 = a3;
  if ( v13 == (__int64 *)(v11 + 16 * v10) )
  {
LABEL_76:
    v15 = a3;
    goto LABEL_12;
  }
  sub_1BCE8C0(a1 + 1192, &v53);
  v16 = sub_1BBA1B0(a4, a2[v15 - 1]);
  v18 = (__int64 *)sub_1BC93C0(*v17, v16);
  v6 = a2;
  if ( v18 )
  {
    v19 = (__int64 *)v18[2];
    if ( v19 || (__int64 *)v18[1] != v18 )
    {
      v20 = 0;
      while ( 1 )
      {
        if ( v18[13] == *v18 )
          v20 = *v18;
        if ( !v19 )
          break;
        v18 = v19;
        v19 = (__int64 *)v19[2];
      }
      v21 = v53;
      if ( v20 )
        goto LABEL_50;
    }
  }
LABEL_12:
  v22 = (__int64 *)v59;
  v23 = &a2[v15];
  v54 = 0;
  v55 = (__int64 *)v59;
  v56 = (__int64 *)v59;
  v57 = 16;
  v58 = 0;
  if ( a2 == v23 )
  {
    v21 = v53;
    v24 = (__int64 *)v59;
    v20 = 0;
    v29 = v50 + 24;
    v52 = v53 + 40;
    if ( v53 + 40 == v50 + 24 )
      goto LABEL_50;
LABEL_26:
    v20 = 0;
    while ( 1 )
    {
      v30 = 0;
      if ( v29 )
        v30 = v29 - 24;
      if ( v22 == v24 )
      {
        v31 = HIDWORD(v57);
        v32 = &v22[v31];
        if ( &v22[v31] == v22 )
        {
LABEL_66:
          v22 = &v24[v31];
          v33 = &v24[v31];
        }
        else
        {
          while ( v30 != *v22 )
          {
            if ( v32 == ++v22 )
              goto LABEL_66;
          }
          v33 = &v24[v31];
        }
      }
      else
      {
        v22 = sub_16CC9F0((__int64)&v54, v30);
        v24 = v56;
        if ( v30 == *v22 )
        {
          if ( v56 == v55 )
            v33 = &v56[HIDWORD(v57)];
          else
            v33 = &v56[(unsigned int)v57];
        }
        else
        {
          if ( v56 != v55 )
            goto LABEL_29;
          v22 = &v56[HIDWORD(v57)];
          v33 = v22;
        }
      }
      if ( v33 == v22 )
      {
        v24 = v56;
LABEL_29:
        v22 = v55;
        if ( HIDWORD(v57) == v58 )
          goto LABEL_47;
        goto LABEL_30;
      }
      *v22 = -2;
      v34 = a4[1];
      v35 = v58 + 1;
      v36 = 0;
      ++v58;
      if ( v34 )
        v36 = *(unsigned __int8 *)(v34 + 16) - 24;
      v37 = *(unsigned __int8 *)(v30 + 16) - 24;
      if ( v37 == v36 )
        goto LABEL_65;
      v38 = a4[2];
      v39 = 0;
      if ( v38 )
        v39 = *(unsigned __int8 *)(v38 + 16) - 24;
      if ( v37 == v39 )
LABEL_65:
        v20 = v30;
      v24 = v56;
      v22 = v55;
      if ( HIDWORD(v57) == v35 )
        goto LABEL_47;
LABEL_30:
      v29 = *(_QWORD *)(v29 + 8);
      if ( v52 == v29 )
        goto LABEL_47;
    }
  }
  v24 = (__int64 *)v59;
  do
  {
LABEL_16:
    v25 = *v6;
    if ( v24 != v22 )
    {
LABEL_14:
      v51 = v6;
      sub_16CCBA0((__int64)&v54, *v6);
      v24 = v56;
      v22 = v55;
      v6 = v51;
      goto LABEL_15;
    }
    v26 = &v24[HIDWORD(v57)];
    if ( v26 == v24 )
    {
LABEL_72:
      if ( HIDWORD(v57) >= (unsigned int)v57 )
        goto LABEL_14;
      ++HIDWORD(v57);
      *v26 = v25;
      v22 = v55;
      ++v54;
      v24 = v56;
    }
    else
    {
      v27 = v24;
      v28 = 0;
      while ( v25 != *v27 )
      {
        if ( *v27 == -2 )
          v28 = v27;
        if ( v26 == ++v27 )
        {
          if ( !v28 )
            goto LABEL_72;
          ++v6;
          *v28 = v25;
          v24 = v56;
          --v58;
          v22 = v55;
          ++v54;
          if ( v23 != v6 )
            goto LABEL_16;
          goto LABEL_25;
        }
      }
    }
LABEL_15:
    ++v6;
  }
  while ( v23 != v6 );
LABEL_25:
  v29 = v50 + 24;
  v52 = v53 + 40;
  if ( v50 + 24 != v53 + 40 )
    goto LABEL_26;
  v20 = 0;
LABEL_47:
  if ( v24 != v22 )
    _libc_free((unsigned __int64)v24);
  v21 = v53;
LABEL_50:
  v40 = *(_QWORD *)(v20 + 32);
  *(_QWORD *)(a1 + 1408) = v21;
  *(_QWORD *)(a1 + 1416) = v40;
  if ( v40 != v21 + 40 )
  {
    if ( !v40 )
      BUG();
    v41 = *(_QWORD *)(v40 + 24);
    v54 = (unsigned __int8 *)v41;
    if ( v41 )
    {
      sub_1623A60((__int64)&v54, v41, 2);
      v42 = *(_QWORD *)(a1 + 1400);
      if ( v42 )
        goto LABEL_54;
LABEL_55:
      v43 = v54;
      *(_QWORD *)(a1 + 1400) = v54;
      if ( v43 )
        sub_1623210((__int64)&v54, v43, a1 + 1400);
    }
    else
    {
      v42 = *(_QWORD *)(a1 + 1400);
      if ( v42 )
      {
LABEL_54:
        sub_161E7C0(a1 + 1400, v42);
        goto LABEL_55;
      }
    }
  }
  result = v50;
  v45 = *(_QWORD *)(v50 + 48);
  v54 = (unsigned __int8 *)v45;
  if ( v45 )
  {
    result = sub_1623A60((__int64)&v54, v45, 2);
    v46 = *(_QWORD *)(a1 + 1400);
    if ( v46 )
      goto LABEL_59;
LABEL_60:
    v47 = v54;
    *(_QWORD *)(a1 + 1400) = v54;
    if ( v47 )
      return sub_1623210((__int64)&v54, v47, a1 + 1400);
  }
  else
  {
    v46 = *(_QWORD *)(a1 + 1400);
    if ( v46 )
    {
LABEL_59:
      result = sub_161E7C0(a1 + 1400, v46);
      goto LABEL_60;
    }
  }
  return result;
}
