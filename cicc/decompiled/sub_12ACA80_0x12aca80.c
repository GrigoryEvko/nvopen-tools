// Function: sub_12ACA80
// Address: 0x12aca80
//
__int64 __fastcall sub_12ACA80(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int v6; // r14d
  __int64 *v7; // r12
  __int64 v8; // r15
  __int64 v9; // r13
  int v10; // r13d
  int v11; // r15d
  int v12; // r8d
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r15
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // r12d
  __int64 v22; // rax
  __int64 v23; // r15
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // r15
  __int64 v29; // rdi
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rsi
  _QWORD *v33; // rdx
  _QWORD *v34; // rdi
  __int64 v36; // rax
  __int64 v37; // rdx
  _DWORD *v39; // [rsp+18h] [rbp-1E8h]
  __int64 v40; // [rsp+18h] [rbp-1E8h]
  char *v41; // [rsp+20h] [rbp-1E0h]
  char *v42; // [rsp+28h] [rbp-1D8h]
  char *v43; // [rsp+30h] [rbp-1D0h]
  __int64 v44; // [rsp+30h] [rbp-1D0h]
  __int64 v45; // [rsp+30h] [rbp-1D0h]
  char *v46; // [rsp+38h] [rbp-1C8h]
  __int64 *v47; // [rsp+40h] [rbp-1C0h]
  char v48; // [rsp+48h] [rbp-1B8h]
  int v49; // [rsp+48h] [rbp-1B8h]
  char v50; // [rsp+4Ch] [rbp-1B4h]
  int v51; // [rsp+4Ch] [rbp-1B4h]
  __int64 *v52; // [rsp+50h] [rbp-1B0h]
  __int64 v53; // [rsp+50h] [rbp-1B0h]
  __int64 *v54; // [rsp+58h] [rbp-1A8h]
  __int64 v55; // [rsp+58h] [rbp-1A8h]
  unsigned __int64 *v56; // [rsp+58h] [rbp-1A8h]
  unsigned int v57; // [rsp+64h] [rbp-19Ch] BYREF
  __int64 v58; // [rsp+68h] [rbp-198h] BYREF
  _QWORD v59[2]; // [rsp+70h] [rbp-190h] BYREF
  char v60[16]; // [rsp+80h] [rbp-180h] BYREF
  __int16 v61; // [rsp+90h] [rbp-170h]
  _BYTE v62[16]; // [rsp+A0h] [rbp-160h] BYREF
  __int16 v63; // [rsp+B0h] [rbp-150h]
  _QWORD *v64; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-138h]
  _QWORD v66[38]; // [rsp+D0h] [rbp-130h] BYREF

  v4 = (unsigned int)(a3 - 678);
  if ( (unsigned int)v4 > 0x1D )
  {
    v36 = (unsigned int)(a3 - 708);
    if ( (unsigned int)v36 > 0x17 )
    {
      v37 = (unsigned int)(a3 - 732);
      if ( (unsigned int)v37 > 0xC )
        sub_127B630("unexpected WMMA intrinsic!", 0);
      v48 = 1;
      v50 = 0;
      v6 = dword_4281020[v37];
    }
    else
    {
      v48 = 0;
      v50 = 0;
      v6 = dword_4281060[v36];
    }
  }
  else
  {
    v48 = 0;
    v50 = 1;
    v6 = dword_42810C0[v4];
  }
  v39 = (_DWORD *)(a4 + 36);
  v7 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL) + 16LL);
  v47 = *(__int64 **)(*(_QWORD *)(a4 + 72) + 16LL);
  v54 = (__int64 *)v7[2];
  v8 = *(_QWORD *)(v54[2] + 16);
  v52 = (__int64 *)v54[2];
  v9 = *(_QWORD *)(v8 + 16);
  sub_12A6F10(v8, 3u, "unexpected 'rowcol' operand", "'rowcol' operand can be 0, 1, 2, or 3 only", (_DWORD *)(a4 + 36));
  v46 = sub_128F980((__int64)a2, (__int64)v47);
  v43 = sub_128F980((__int64)a2, (__int64)v7);
  v42 = sub_128F980((__int64)a2, (__int64)v54);
  v41 = sub_128F980((__int64)a2, (__int64)v52);
  v66[0] = sub_128F980((__int64)a2, v8);
  v64 = v66;
  v65 = 0x2000000001LL;
  if ( v6 != 3752 )
  {
    sub_12A6F10(v9, 1u, "unexpected 'satf' operand", "'satf' operand can be 0, or 1 only", v39);
    v64[(unsigned int)v65] = sub_128F980((__int64)a2, v9);
    LODWORD(v65) = v65 + 1;
    if ( !v50 )
    {
      if ( v48 )
        goto LABEL_6;
      if ( v6 > 0xFB4 )
      {
        if ( v6 - 4027 <= 1 )
        {
          v51 = 8;
          v10 = 8;
          v11 = 4;
          v12 = 1;
          goto LABEL_7;
        }
        goto LABEL_43;
      }
      if ( v6 > 0xFB2 )
      {
        v51 = 8;
        v10 = 8;
        v11 = 1;
        v12 = 4;
        goto LABEL_7;
      }
LABEL_21:
      if ( v6 - 4011 <= 1 )
      {
        v51 = 8;
        v10 = 8;
        v11 = 2;
        v12 = 2;
        goto LABEL_7;
      }
LABEL_43:
      sub_127B630("unexpected imma_mma intrinsic call!", 0);
    }
    if ( ((v6 - 3967) & 0xFFFFFFFD) == 0 || (v6 & 0xFFFFFFFD) == 0xF8D )
      goto LABEL_27;
    goto LABEL_38;
  }
  if ( v50 )
  {
LABEL_38:
    v10 = 4;
    if ( ((v6 - 3991) & 0xFFFFFFFD) != 0 )
    {
LABEL_28:
      if ( v6 - 3966 > 0x19 )
      {
        v51 = 8;
        v11 = 8;
        v12 = 8;
      }
      else
      {
        v11 = 8;
        v12 = 8;
        v51 = ((0x300C003uLL >> ((unsigned __int8)v6 - 126)) & 1) != 0 ? 4 : 8;
      }
      goto LABEL_7;
    }
LABEL_27:
    v10 = 8;
    goto LABEL_28;
  }
  if ( !v48 )
    goto LABEL_21;
LABEL_6:
  v51 = 2;
  v10 = 2;
  v11 = 1;
  v12 = 1;
LABEL_7:
  v13 = a2[4];
  v59[1] = &v64;
  v59[0] = a2;
  v49 = v12;
  v40 = v13 + 8;
  v14 = sub_8D46C0(*v7);
  v15 = sub_127A030(v40, v14, 0);
  sub_12A8A80(v59, v49, v43, v15);
  v44 = a2[4] + 8LL;
  v16 = sub_8D46C0(*v54);
  v17 = sub_127A030(v44, v16, 0);
  sub_12A8A80(v59, v11, v42, v17);
  v18 = a2[4] + 8LL;
  v19 = sub_8D46C0(*v52);
  v20 = sub_127A030(v18, v19, 0);
  v21 = 0;
  sub_12A8A80(v59, v10, v41, v20);
  v22 = sub_126A190((_QWORD *)a2[4], v6, 0, 0);
  v63 = 257;
  v45 = sub_1285290(a2 + 6, *(_QWORD *)(v22 + 24), v22, (int)v64, (unsigned int)v65, (__int64)v62, 0);
  do
  {
    v63 = 257;
    v23 = a2[4] + 8LL;
    v24 = sub_8D46C0(*v47);
    v25 = sub_127A030(v23, v24, 0);
    v26 = sub_12A8800(a2 + 6, v25, v46, v21, (__int64)v62);
    v61 = 257;
    v53 = v26;
    v57 = v21;
    v55 = sub_12A9E60(a2 + 6, v45, (__int64)&v57, 1, (__int64)v60);
    v63 = 257;
    v27 = sub_1648A60(64, 2);
    v28 = (_QWORD *)v27;
    if ( v27 )
      sub_15F9650(v27, v55, v53, 0, 0);
    v29 = a2[7];
    if ( v29 )
    {
      v56 = (unsigned __int64 *)a2[8];
      sub_157E9D0(v29 + 40, v28);
      v30 = *v56;
      v31 = v28[3] & 7LL;
      v28[4] = v56;
      v30 &= 0xFFFFFFFFFFFFFFF8LL;
      v28[3] = v30 | v31;
      *(_QWORD *)(v30 + 8) = v28 + 3;
      *v56 = *v56 & 7 | (unsigned __int64)(v28 + 3);
    }
    sub_164B780(v28, v62);
    v32 = a2[6];
    if ( v32 )
    {
      v58 = a2[6];
      sub_1623A60(&v58, v32, 2);
      v33 = v28 + 6;
      if ( v28[6] )
      {
        sub_161E7C0(v28 + 6);
        v33 = v28 + 6;
      }
      v32 = v58;
      v28[6] = v58;
      if ( v32 )
        sub_1623210(&v58, v32, v33);
    }
    ++v21;
  }
  while ( v51 != v21 );
  v34 = v64;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v34 != v66 )
    _libc_free(v34, v32);
  return a1;
}
