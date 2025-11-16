// Function: sub_1CC64A0
// Address: 0x1cc64a0
//
__int64 __fastcall sub_1CC64A0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rdx
  __int64 *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  _QWORD *v13; // rcx
  unsigned int v14; // eax
  unsigned __int64 v15; // rax
  _QWORD *v16; // r12
  __int64 v17; // r13
  unsigned __int8 v18; // al
  __int64 v19; // rax
  _BYTE *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r9
  _QWORD *v24; // rsi
  __int64 v25; // r14
  _BOOL4 v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // r8d
  __int64 v30; // r9
  bool v31; // al
  _QWORD *v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // r15
  __int64 *v36; // rcx
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // r15
  __int64 *v40; // rsi
  __int64 *v41; // rcx
  unsigned int i; // r15d
  __int64 *v43; // r14
  __int64 *v44; // rax
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 *v49; // rdx
  __int64 v51; // rax
  unsigned __int8 v52; // [rsp+Fh] [rbp-981h]
  __int64 v53; // [rsp+18h] [rbp-978h]
  int v54; // [rsp+18h] [rbp-978h]
  __int64 v55; // [rsp+20h] [rbp-970h] BYREF
  __int64 *v56; // [rsp+28h] [rbp-968h]
  __int64 *v57; // [rsp+30h] [rbp-960h]
  __int64 v58; // [rsp+38h] [rbp-958h]
  int v59; // [rsp+40h] [rbp-950h]
  _BYTE v60[264]; // [rsp+48h] [rbp-948h] BYREF
  _QWORD *v61; // [rsp+150h] [rbp-840h] BYREF
  unsigned int v62; // [rsp+158h] [rbp-838h]
  unsigned int v63; // [rsp+15Ch] [rbp-834h]
  _QWORD v64[262]; // [rsp+160h] [rbp-830h] BYREF

  v9 = (__int64 *)v60;
  v10 = (__int64 *)v60;
  v11 = *(_QWORD *)(a1 + 80);
  v61 = v64;
  v12 = v11 - 24;
  if ( !v11 )
    v12 = 0;
  v56 = (__int64 *)v60;
  v13 = v64;
  v55 = 0;
  v64[0] = v12;
  v14 = 1;
  v57 = (__int64 *)v60;
  v58 = 32;
  v59 = 0;
  v63 = 256;
  v62 = 1;
  v52 = 0;
  while ( 1 )
  {
    v39 = v13[v14 - 1];
    v62 = v14 - 1;
    if ( v10 != v9 )
      goto LABEL_4;
    v40 = &v9[HIDWORD(v58)];
    if ( v40 == v9 )
      goto LABEL_59;
    v41 = 0;
    do
    {
      if ( v39 == *v9 )
        goto LABEL_5;
      if ( *v9 == -2 )
        v41 = v9;
      ++v9;
    }
    while ( v40 != v9 );
    if ( !v41 )
    {
LABEL_59:
      if ( HIDWORD(v58) >= (unsigned int)v58 )
      {
LABEL_4:
        sub_16CCBA0((__int64)&v55, v39);
      }
      else
      {
        ++HIDWORD(v58);
        *v40 = v39;
        ++v55;
      }
LABEL_5:
      v15 = sub_157EBA0(v39);
      v16 = (_QWORD *)v15;
      if ( !v15 )
        goto LABEL_27;
      goto LABEL_6;
    }
    *v41 = v39;
    --v59;
    ++v55;
    v15 = sub_157EBA0(v39);
    v16 = (_QWORD *)v15;
    if ( !v15 )
      goto LABEL_27;
LABEL_6:
    if ( *(_BYTE *)(v15 + 16) == 26 && (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) == 3 )
    {
      v17 = *(v16 - 9);
      v18 = *(_BYTE *)(v17 + 16);
      if ( v18 <= 0x17u )
      {
        if ( v18 != 5 )
          goto LABEL_27;
        v20 = (_BYTE *)sub_1632FA0(*(_QWORD *)(a1 + 40));
        v23 = sub_14DBA30(v17, (__int64)v20, 0);
      }
      else
      {
        v19 = sub_15F2050(*(v16 - 9));
        v20 = (_BYTE *)sub_1632FA0(v19);
        v23 = sub_14DD210((__int64 *)v17, v20, 0);
      }
      if ( v23 && *(_BYTE *)(v23 + 16) == 13 )
      {
        v53 = v23;
        v24 = (_QWORD *)v16[-3 * (unsigned __int8)sub_1595F50(v23, (__int64)v20, v21, v22) - 3];
        v25 = (__int64)v24;
        v26 = sub_183E920((__int64)&v55, (__int64)v24);
        v30 = v53;
        if ( !v26 )
        {
          v51 = v62;
          if ( v62 >= v63 )
          {
            v24 = v64;
            sub_16CD150((__int64)&v61, v64, 0, 8, v29, v53);
            v51 = v62;
            v30 = v53;
          }
          v27 = (__int64)v61;
          v61[v51] = v25;
          ++v62;
        }
        v31 = sub_15962C0(v30, (__int64)v24, v27, v28);
        sub_157F2D0(v16[-3 * v31 - 3], v39, 0);
        v32 = sub_1648A60(56, 1u);
        v35 = (__int64)v32;
        if ( v32 )
          sub_15F8320((__int64)v32, v25, (__int64)v16);
        sub_164D160((__int64)v16, v35, a2, a3, a4, a5, v33, v34, a8, a9);
        sub_15F20C0(v16);
        sub_1AEB370(v17, 0);
        v52 = 1;
        goto LABEL_16;
      }
    }
LABEL_27:
    v54 = sub_15F3BE0((__int64)v16);
    if ( v54 )
    {
      for ( i = 0; v54 != i; ++i )
      {
        v47 = sub_15F3BF0((__int64)v16, i);
        v44 = v56;
        if ( v57 == v56 )
        {
          v43 = &v56[HIDWORD(v58)];
          if ( v56 == v43 )
          {
            v49 = v56;
          }
          else
          {
            do
            {
              if ( v47 == *v44 )
                break;
              ++v44;
            }
            while ( v43 != v44 );
            v49 = &v56[HIDWORD(v58)];
          }
LABEL_42:
          while ( v49 != v44 )
          {
            if ( (unsigned __int64)*v44 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_32;
            ++v44;
          }
          if ( v44 != v43 )
            continue;
        }
        else
        {
          v43 = &v57[(unsigned int)v58];
          v44 = sub_16CC9F0((__int64)&v55, v47);
          if ( v47 == *v44 )
          {
            if ( v57 == v56 )
              v49 = &v57[HIDWORD(v58)];
            else
              v49 = &v57[(unsigned int)v58];
            goto LABEL_42;
          }
          if ( v57 == v56 )
          {
            v44 = &v57[HIDWORD(v58)];
            v49 = v44;
            goto LABEL_42;
          }
          v44 = &v57[(unsigned int)v58];
LABEL_32:
          if ( v44 != v43 )
            continue;
        }
        v48 = v62;
        if ( v62 >= v63 )
        {
          sub_16CD150((__int64)&v61, v64, 0, 8, v45, v46);
          v48 = v62;
        }
        v61[v48] = v47;
        ++v62;
      }
    }
LABEL_16:
    v14 = v62;
    if ( !v62 )
      break;
    v13 = v61;
    v10 = v57;
    v9 = v56;
  }
  if ( v52 )
    sub_1AF0CE0(a1, 0, 0, v36, a2, a3, a4, a5, v37, v38, a8, a9);
  if ( v61 != v64 )
    _libc_free((unsigned __int64)v61);
  if ( v57 != v56 )
    _libc_free((unsigned __int64)v57);
  return v52;
}
