// Function: sub_169B680
// Address: 0x169b680
//
__int64 __fastcall sub_169B680(__int16 **a1, __int64 a2, unsigned int a3, int a4, unsigned int a5)
{
  __int16 *i; // rax
  int v6; // edi
  unsigned int v7; // ebx
  __int64 v8; // rcx
  unsigned int v9; // ebx
  unsigned int v10; // r11d
  __int16 **v11; // rax
  __int16 **v12; // r14
  __int16 **v13; // r13
  _DWORD *v14; // r15
  unsigned int v15; // r12d
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int16 **v18; // rdx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned int v21; // r12d
  int v22; // ebx
  int v23; // ebx
  bool v24; // dl
  _BOOL4 v25; // eax
  int v26; // ebx
  unsigned __int64 v27; // rbx
  unsigned __int64 *v28; // rdx
  unsigned int v29; // r8d
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rbx
  unsigned int v33; // r15d
  __int64 v34; // rax
  unsigned int v35; // ebx
  __int64 v36; // r12
  unsigned int v37; // eax
  unsigned int v38; // edx
  unsigned int v39; // r12d
  __int16 **v41; // rdx
  int v42; // eax
  unsigned __int64 *v43; // rax
  unsigned __int64 *v44; // rax
  int v45; // eax
  unsigned int v49; // [rsp+24h] [rbp-4BDCh]
  unsigned int v50; // [rsp+28h] [rbp-4BD8h]
  __int16 **v52; // [rsp+38h] [rbp-4BC8h]
  __int16 **v53; // [rsp+38h] [rbp-4BC8h]
  int v54; // [rsp+38h] [rbp-4BC8h]
  unsigned int v55; // [rsp+40h] [rbp-4BC0h]
  unsigned int v56; // [rsp+40h] [rbp-4BC0h]
  unsigned int v57; // [rsp+40h] [rbp-4BC0h]
  unsigned int v58; // [rsp+44h] [rbp-4BBCh]
  unsigned int v59; // [rsp+44h] [rbp-4BBCh]
  __int64 v61; // [rsp+54h] [rbp-4BACh] BYREF
  int v62; // [rsp+5Ch] [rbp-4BA4h]
  _OWORD v63[4]; // [rsp+60h] [rbp-4BA0h] BYREF
  _QWORD v64[600]; // [rsp+A0h] [rbp-4B60h] BYREF
  __int16 *v65[2]; // [rsp+1360h] [rbp-38A0h] BYREF
  __int16 v66; // [rsp+1370h] [rbp-3890h]
  __int16 *v67[1212]; // [rsp+2620h] [rbp-25E0h] BYREF

  v49 = a5 & 0xFFFFFFFB;
  i = *a1;
  v61 = 2147581951LL;
  v62 = 0;
  v6 = *((_DWORD *)i + 1);
  memset(v63, 0, sizeof(v63));
  v58 = (unsigned int)(v6 + 74) >> 6;
  v7 = abs32(a4);
  v67[0] = (__int16 *)390625;
  v8 = v7 & 7;
  v9 = v7 >> 3;
  LODWORD(v63[0]) = 1;
  v64[0] = qword_42AE940[v8];
  if ( !v9 )
  {
    v50 = 1;
    goto LABEL_19;
  }
  v10 = 1;
  v11 = v65;
  v12 = (__int16 **)v64;
  v13 = v67;
  v14 = (_DWORD *)v63 + 1;
  v15 = v9;
  v16 = 1;
  while ( 1 )
  {
    if ( (v15 & 1) == 0 )
      goto LABEL_3;
    v56 = v10;
    v53 = v11;
    sub_16A7C60(v11, v12, v13, v10, (unsigned int)v16);
    v10 = v16 + v56;
    if ( !(&v53[(unsigned int)v16 - 1])[v56] )
      break;
    v41 = v12;
    v12 = v53;
    v11 = v41;
LABEL_3:
    v15 >>= 1;
    v13 += v16;
    if ( !v15 )
      goto LABEL_12;
LABEL_4:
    v16 = (unsigned int)*v14;
    if ( !(_DWORD)v16 )
    {
      v17 = (unsigned int)*(v14 - 1);
      v55 = v10;
      v52 = v11;
      v16 = (unsigned int)(2 * v17);
      sub_16A7C60(v13, &v13[-v17], &v13[-v17], (unsigned int)v17, (unsigned int)v17);
      v10 = v55;
      v11 = v52;
      if ( !v13[(unsigned int)(v16 - 1)] )
        v16 = (unsigned int)(v16 - 1);
      *v14 = v16;
    }
    ++v14;
  }
  v18 = v12;
  v15 >>= 1;
  v12 = v53;
  v10 = v16 + v56 - 1;
  v11 = v18;
  v13 += v16;
  if ( v15 )
    goto LABEL_4;
LABEL_12:
  v50 = v10;
  if ( v12 != v64 )
    sub_16A7050(v64, v12, v10);
  for ( i = *a1; ; i = *a1 )
  {
LABEL_19:
    HIDWORD(v61) = (v58 << 6) - 1;
    v21 = HIDWORD(v61) - *((_DWORD *)i + 1);
    sub_1698390((__int64)v65, (__int64)&v61);
    sub_169B620((__int64)v65, (*((_BYTE *)a1 + 18) & 8) != 0);
    sub_1698360((__int64)v67, (__int64)&v61);
    v54 = sub_169A140(v65, a2, a3, 0);
    v22 = sub_169A140(v67, (__int64)v64, v50, 0);
    v66 += a4;
    if ( a4 < 0 )
    {
      v42 = sub_1698970(v65, (__int64)v67);
      v57 = v21;
      if ( v66 < (*a1)[1] )
      {
        v57 = (*a1)[1] - v66 + v21;
        v21 = v57;
        if ( HIDWORD(v61) <= v57 )
          v21 = HIDWORD(v61);
      }
      v23 = v42 | v22;
      if ( v23 )
      {
        v24 = v42 != 0;
        v26 = 2 - ((v54 == 0) - 1);
        goto LABEL_35;
      }
      v25 = 0;
      v24 = 0;
    }
    else
    {
      v57 = v21;
      v23 = v22 != 0;
      v24 = (unsigned int)sub_16999D0((__int64)v65, (__int64)v67, 0) != 0;
      v25 = v24;
    }
    v26 = v23 - ((v54 == 0) - 1);
    if ( !v26 )
    {
      v27 = (unsigned int)(2 * v25);
      goto LABEL_23;
    }
LABEL_35:
    v27 = (unsigned int)v24 + 2 * v26;
LABEL_23:
    v28 = (unsigned __int64 *)sub_1698470((__int64)v65);
    v29 = (v21 - 1) >> 6;
    v30 = v28[v29] & (0xFFFFFFFFFFFFFFFFLL >> (63 - ((v21 - 1) & 0x3F)));
    v31 = 1LL << ((v21 - 1) & 0x3F);
    if ( v49 )
      v31 = 0;
    if ( v21 - 1 <= 0x3F )
    {
      v19 = v30 - v31;
      v20 = v31 - v30;
      if ( v19 <= v20 )
        v20 = v19;
      goto LABEL_17;
    }
    if ( v30 != v31 )
      break;
    v43 = &v28[v29 - 1];
    while ( v28 != v43 )
    {
      --v43;
      if ( v43[1] )
        goto LABEL_28;
    }
    v20 = *v28;
LABEL_17:
    if ( v27 <= 2 * v20 )
      goto LABEL_28;
    sub_1698460((__int64)v67);
    sub_1698460((__int64)v65);
    v58 *= 2;
  }
  if ( v30 == v31 - 1 )
  {
    v44 = &v28[v29 - 1];
    while ( v28 != v44 )
    {
      --v44;
      if ( v44[1] != -1 )
        goto LABEL_28;
    }
    v20 = -(__int64)*v28;
    goto LABEL_17;
  }
LABEL_28:
  v59 = HIDWORD(v61) - v21;
  v32 = sub_1698470((__int64)v65);
  v33 = sub_1698310((__int64)a1);
  v34 = sub_1698470((__int64)a1);
  sub_16A8750(v34, v33, v32, v59, v21);
  *((_WORD *)a1 + 8) = (*a1)[2] + v66 - WORD2(v61) + v21;
  v35 = sub_1698310((__int64)v65);
  v36 = sub_1698470((__int64)v65);
  v37 = sub_16A7110(v36, v35);
  v38 = 0;
  if ( v37 < v57 )
  {
    if ( v37 + 1 == v57 )
    {
      v38 = 2;
    }
    else if ( v35 << 6 < v57 || (v45 = sub_16A70B0(v36, v57 - 1), v38 = 3, !v45) )
    {
      v38 = 1;
    }
  }
  v39 = sub_1698EC0(a1, a5, v38);
  sub_1698460((__int64)v67);
  sub_1698460((__int64)v65);
  return v39;
}
