// Function: sub_F1E7F0
// Address: 0xf1e7f0
//
__int64 __fastcall sub_F1E7F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // bl
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // r14d
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rdi
  unsigned __int64 *v16; // rax
  unsigned __int8 v17; // dl
  unsigned __int8 v18; // si
  char v19; // dh
  char v20; // dl
  __int64 v21; // rcx
  unsigned __int8 v22; // cl
  __int64 v23; // r10
  __int64 v24; // rsi
  unsigned __int8 v25; // cl
  bool v26; // r13
  unsigned __int8 *v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // eax
  __int64 *j; // rdx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 *v37; // r9
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rsi
  unsigned int v41; // ecx
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // r13
  __int64 v46; // r14
  __int64 v47; // r12
  unsigned int i; // ebx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rax
  int v55; // edx
  int v56; // r9d
  __int64 v57; // [rsp+0h] [rbp-130h]
  __int64 v58; // [rsp+8h] [rbp-128h]
  __int64 v59; // [rsp+8h] [rbp-128h]
  __int64 v60; // [rsp+10h] [rbp-120h]
  bool v61; // [rsp+18h] [rbp-118h]
  unsigned __int8 v62; // [rsp+18h] [rbp-118h]
  __int64 v63; // [rsp+20h] [rbp-110h]
  __int64 v64; // [rsp+28h] [rbp-108h] BYREF
  __int64 v65[2]; // [rsp+30h] [rbp-100h] BYREF
  __m128i v66; // [rsp+40h] [rbp-F0h] BYREF
  bool v67; // [rsp+70h] [rbp-C0h]
  __int64 *v68; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v69; // [rsp+88h] [rbp-A8h]
  _BYTE v70[48]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 *v71; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v72; // [rsp+C8h] [rbp-68h]
  __int64 v73; // [rsp+D0h] [rbp-60h] BYREF
  int v74; // [rsp+D8h] [rbp-58h]
  char v75; // [rsp+DCh] [rbp-54h]
  _BYTE v76[80]; // [rsp+E0h] [rbp-50h] BYREF

  v3 = *(_BYTE *)a2;
  v64 = a3;
  if ( v3 == 84 )
    return 0;
  v4 = a1;
  v5 = a2;
  v6 = (unsigned int)v3 - 39;
  if ( (unsigned int)v6 <= 0x38 )
  {
    v7 = 0x100060000000001LL;
    if ( _bittest64(&v7, v6) )
      return 0;
  }
  if ( (unsigned __int8)sub_B46790((unsigned __int8 *)a2, 0) )
    return 0;
  v8 = sub_B46900((unsigned __int8 *)a2);
  if ( !(_BYTE)v8 || v3 == 60 || (unsigned int)v3 - 30 <= 0xA )
    return 0;
  v9 = *(_QWORD *)(v64 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v64 + 48 )
    goto LABEL_83;
  if ( !v9 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_83:
    BUG();
  if ( *(_BYTE *)(v9 - 24) == 39 )
    return 0;
  v63 = *(_QWORD *)(a2 + 40);
  if ( v3 == 79 )
  {
    if ( (unsigned __int8)sub_DF9980(*(_QWORD *)(a1 + 8)) )
    {
      v38 = *(_QWORD *)(a1 + 64);
      if ( !*(_BYTE *)(v38 + 192) )
        sub_CFDFC0(*(_QWORD *)(a1 + 64), 0, v34, v35, v36, v37);
      v39 = *(unsigned int *)(v38 + 184);
      if ( (_DWORD)v39 )
      {
        v40 = *(_QWORD *)(v38 + 168);
        v41 = (v39 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v42 = v40 + 88LL * v41;
        v43 = *(_QWORD *)(v42 + 24);
        if ( v5 == v43 )
        {
LABEL_57:
          if ( v42 != v40 + 88 * v39 )
          {
            v44 = *(_QWORD *)(v42 + 40);
            if ( v44 + 32LL * *(unsigned int *)(v42 + 48) != v44 )
            {
              v60 = v4;
              v45 = *(_QWORD *)(v42 + 40);
              v59 = v5;
              v62 = v8;
              v46 = v44 + 32LL * *(unsigned int *)(v42 + 48);
              do
              {
                v47 = *(_QWORD *)(v45 + 16);
                if ( v47 )
                {
                  for ( i = 0; *(char *)(v47 + 7) < 0; ++i )
                  {
                    v49 = sub_BD2BC0(v47);
                    v51 = v49 + v50;
                    v52 = 0;
                    if ( *(char *)(v47 + 7) < 0 )
                      v52 = sub_BD2BC0(v47);
                    if ( i >= (unsigned int)((v51 - v52) >> 4) )
                      break;
                    v53 = 0;
                    if ( *(char *)(v47 + 7) < 0 )
                      v53 = sub_BD2BC0(v47);
                    v54 = *(_QWORD *)(v53 + 16LL * i);
                    if ( *(_QWORD *)v54 == 5 && *(_DWORD *)(v54 + 16) == 1734962273 && *(_BYTE *)(v54 + 20) == 110 )
                      return 0;
                  }
                }
                v45 += 32;
              }
              while ( v45 != v46 );
              v8 = v62;
              v4 = v60;
              v5 = v59;
            }
          }
        }
        else
        {
          v55 = 1;
          while ( v43 != -4096 )
          {
            v56 = v55 + 1;
            v41 = (v39 - 1) & (v55 + v41);
            v42 = v40 + 88LL * v41;
            v43 = *(_QWORD *)(v42 + 24);
            if ( v5 == v43 )
              goto LABEL_57;
            v55 = v56;
          }
        }
      }
    }
    v3 = *(_BYTE *)v5;
  }
  if ( v3 == 85 && ((unsigned __int8)sub_A73ED0((_QWORD *)(v5 + 72), 6) || (unsigned __int8)sub_B49560(v5, 6)) )
    return 0;
  if ( (unsigned __int8)sub_B46490(v5) )
  {
    v25 = *(_BYTE *)v5 - 34;
    if ( v25 > 0x33u )
      return 0;
    v61 = ((0x8000000000041uLL >> v25) & 1) == 0;
    if ( (~(0x8000000000041uLL >> v25) & 1) != 0 )
      return 0;
    sub_D67230(&v66, (unsigned __int8 *)v5, *(__int64 **)(v4 + 72));
    v26 = v67;
    if ( !v67 )
      return 0;
    v27 = sub_98ACB0((unsigned __int8 *)v66.m128i_i64[0], 6u);
    if ( *v27 != 60 )
      return 0;
    v71 = 0;
    v65[0] = (__int64)&v71;
    v69 = 0x600000000LL;
    v65[1] = (__int64)&v68;
    v68 = (__int64 *)v70;
    v72 = (__int64)v76;
    v73 = 4;
    v74 = 0;
    v75 = 1;
    sub_F081B0(v65, (__int64)v27, v65, (unsigned __int64)v76, v28, v29);
    v32 = v69;
    for ( j = v65; (_DWORD)v69; v32 = v69 )
    {
      v27 = (unsigned __int8 *)v68[v32 - 1];
      LODWORD(v69) = v32 - 1;
      if ( ((*v27 - 63) & 0xEF) != 0 )
      {
        if ( (unsigned __int8 *)v5 != v27 )
          goto LABEL_45;
      }
      else
      {
        sub_F081B0(v65, (__int64)v27, j, (unsigned __int64)v68, v30, v31);
      }
    }
    v61 = v26;
LABEL_45:
    if ( !v75 )
      _libc_free(v72, v27);
    if ( v68 != (__int64 *)v70 )
      _libc_free(v68, v27);
    if ( !v61 )
      return 0;
  }
  if ( (unsigned __int8)sub_B46420(v5) && ((*(_BYTE *)(v5 + 7) & 0x20) == 0 || !sub_B91C10(v5, 6)) )
  {
    v10 = sub_AA5510(v64);
    v11 = *(_QWORD *)(v5 + 40);
    if ( v11 == v10 )
    {
      v12 = *(_QWORD *)(v5 + 32);
      v13 = v11 + 48;
      if ( v13 == v12 )
        goto LABEL_28;
      while ( 1 )
      {
        v14 = v12 - 24;
        if ( !v12 )
          v14 = 0;
        if ( (unsigned __int8)sub_B46490(v14) )
          break;
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          goto LABEL_28;
      }
    }
    return 0;
  }
LABEL_28:
  v72 = v4;
  v71 = &v64;
  sub_BD5EC0(v5, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_F20750, (__int64)&v71);
  v16 = (unsigned __int64 *)sub_AA5190(v64);
  v18 = v17;
  v20 = v19;
  v57 = (__int64)v16;
  if ( !v16 )
  {
    v18 = 0;
    v20 = 0;
  }
  v21 = v18;
  BYTE1(v21) = v20;
  v58 = v21;
  sub_B44550((_QWORD *)v5, v64, v16, v21);
  v69 = 0x200000000LL;
  v72 = 0x200000000LL;
  v68 = (__int64 *)v70;
  v71 = &v73;
  sub_AE7A50((__int64)&v68, v5, (__int64)&v71);
  v22 = v58;
  v23 = v57;
  if ( (_DWORD)v69 )
  {
    sub_F1C5C0(v4, (unsigned __int8 *)v5, v57, v58, v63, v64, &v68);
    v23 = v57;
    v22 = v58;
  }
  v24 = (unsigned int)v72;
  if ( (_DWORD)v72 )
  {
    v24 = v5;
    sub_F1D400(v4, v5, v23, v22, v63, v64, (__int64)&v71);
  }
  if ( v71 != &v73 )
    _libc_free(v71, v24);
  if ( v68 != (__int64 *)v70 )
    _libc_free(v68, v24);
  return v8;
}
