// Function: sub_10E2E20
// Address: 0x10e2e20
//
__int64 __fastcall sub_10E2E20(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v3; // ebx
  __int64 *v4; // r13
  bool v5; // r15
  unsigned int v6; // ebx
  const void *v7; // r13
  unsigned int v8; // ebx
  unsigned __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 *v19; // rdi
  int v20; // edx
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // r10
  __int64 v24; // r13
  void *v25; // rbx
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r13
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // rcx
  __int64 v33; // r13
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // [rsp+0h] [rbp-110h]
  unsigned int v37; // [rsp+0h] [rbp-110h]
  const void *v38; // [rsp+8h] [rbp-108h]
  __int64 v39; // [rsp+8h] [rbp-108h]
  __int64 v40; // [rsp+8h] [rbp-108h]
  __int64 i; // [rsp+8h] [rbp-108h]
  unsigned int v42; // [rsp+10h] [rbp-100h]
  __int64 v43; // [rsp+10h] [rbp-100h]
  __int64 v44; // [rsp+10h] [rbp-100h]
  __int64 v45; // [rsp+10h] [rbp-100h]
  __int64 v46; // [rsp+10h] [rbp-100h]
  __int64 v47; // [rsp+10h] [rbp-100h]
  char *v49; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v50; // [rsp+28h] [rbp-E8h] BYREF
  const void **v51; // [rsp+30h] [rbp-E0h] BYREF
  const void **v52; // [rsp+38h] [rbp-D8h] BYREF
  __int64 *v53; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v54; // [rsp+48h] [rbp-C8h]
  __int64 *v55; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v56; // [rsp+58h] [rbp-B8h]
  __int64 v57; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v58; // [rsp+68h] [rbp-A8h]
  __int64 v59; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v60; // [rsp+78h] [rbp-98h]
  void *v61; // [rsp+80h] [rbp-90h] BYREF
  __int64 v62; // [rsp+88h] [rbp-88h]
  __int16 v63; // [rsp+A0h] [rbp-70h]
  __int64 *v64; // [rsp+B0h] [rbp-60h] BYREF
  const void ***v65; // [rsp+B8h] [rbp-58h]
  char v66; // [rsp+C0h] [rbp-50h]
  __int16 v67; // [rsp+D0h] [rbp-40h]

  v64 = (__int64 *)&v49;
  v2 = *(_QWORD *)(a2 + 8);
  v65 = &v52;
  v66 = 0;
  if ( (unsigned __int8)sub_10E26A0((__int64)&v64, a2) )
  {
    v66 = 0;
    v64 = &v50;
    v65 = &v51;
    if ( !(unsigned __int8)sub_10E2880((__int64)&v64, v49) )
      return 0;
  }
  else
  {
    v64 = (__int64 *)&v49;
    v65 = &v51;
    v66 = 0;
    if ( !(unsigned __int8)sub_10E2A60((__int64)&v64, a2) )
      return 0;
    v65 = &v52;
    v64 = &v50;
    v66 = 0;
    if ( !(unsigned __int8)sub_10E2C40((__int64)&v64, v49) )
      return 0;
  }
  v54 = *((_DWORD *)v52 + 2);
  if ( v54 > 0x40 )
    sub_C43780((__int64)&v53, v52);
  else
    v53 = (__int64 *)*v52;
  sub_C46A40((__int64)&v53, 1);
  v3 = v54;
  v4 = v53;
  v54 = 0;
  v56 = v3;
  v55 = v53;
  if ( v3 > 0x40 )
  {
    if ( (unsigned int)sub_C44630((__int64)&v55) != 1 )
    {
LABEL_7:
      v5 = 1;
      goto LABEL_8;
    }
  }
  else if ( !v53 || ((unsigned __int64)v53 & ((unsigned __int64)v53 - 1)) != 0 )
  {
    goto LABEL_7;
  }
  LODWORD(v62) = *((_DWORD *)v52 + 2);
  if ( (unsigned int)v62 > 0x40 )
    sub_C43780((__int64)&v61, v52);
  else
    v61 = (void *)*v52;
  sub_C46A40((__int64)&v61, 1);
  v13 = v62;
  LODWORD(v62) = 0;
  LODWORD(v65) = v13;
  v42 = v13;
  v38 = v61;
  v64 = (__int64 *)v61;
  v14 = *((_DWORD *)v51 + 2);
  v58 = v14;
  if ( v14 <= 0x40 )
  {
    v15 = (__int64)*v51;
LABEL_34:
    v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~v15;
    if ( !v14 )
      v16 = 0;
    v57 = v16;
    goto LABEL_37;
  }
  sub_C43780((__int64)&v57, v51);
  v14 = v58;
  if ( v58 <= 0x40 )
  {
    v15 = v57;
    goto LABEL_34;
  }
  sub_C43D10((__int64)&v57);
LABEL_37:
  sub_C46250((__int64)&v57);
  v17 = v58;
  v58 = 0;
  v60 = v17;
  v59 = v57;
  if ( v17 <= 0x40 )
  {
    v5 = v38 != (const void *)v57;
  }
  else
  {
    v36 = v57;
    v5 = !sub_C43C50((__int64)&v59, (const void **)&v64);
    if ( v36 )
    {
      j_j___libc_free_0_0(v36);
      if ( v58 > 0x40 )
      {
        if ( v57 )
          j_j___libc_free_0_0(v57);
      }
    }
  }
  if ( v42 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( (unsigned int)v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
LABEL_8:
  if ( v3 > 0x40 && v4 )
    j_j___libc_free_0_0(v4);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  if ( v5 )
    return 0;
  LODWORD(v62) = *((_DWORD *)v52 + 2);
  if ( (unsigned int)v62 > 0x40 )
    sub_C43780((__int64)&v61, v52);
  else
    v61 = (void *)*v52;
  sub_C46A40((__int64)&v61, 1);
  v6 = v62;
  v7 = v61;
  LODWORD(v62) = 0;
  LODWORD(v65) = v6;
  v64 = (__int64 *)v61;
  if ( v6 > 0x40 )
  {
    v8 = v6 - sub_C444A0((__int64)&v64);
    if ( v7 )
    {
      j_j___libc_free_0_0(v7);
      if ( (unsigned int)v62 > 0x40 )
      {
        if ( v61 )
          j_j___libc_free_0_0(v61);
      }
    }
  }
  else
  {
    v8 = 0;
    if ( v61 )
    {
      _BitScanReverse64(&v9, (unsigned __int64)v61);
      v8 = 64 - (v9 ^ 0x3F);
    }
  }
  v10 = v2;
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v10 = **(_QWORD **)(v2 + 16);
  if ( !(unsigned __int8)sub_F0C790((__int64)a1, *(_DWORD *)(v10 + 8) >> 8, v8) )
    return 0;
  v11 = *((_QWORD *)v49 + 2);
  if ( !v11 )
    return 0;
  if ( *(_QWORD *)(v11 + 8) )
    return 0;
  v18 = *(_QWORD *)(v50 + 16);
  if ( !v18 || *(_QWORD *)(v18 + 8) )
    return 0;
  v19 = (__int64 *)sub_BCD140(*(_QWORD **)v2, v8);
  v20 = *(unsigned __int8 *)(v2 + 8);
  if ( (unsigned int)(v20 - 17) <= 1 )
  {
    v21 = *(_DWORD *)(v2 + 32);
    BYTE4(v59) = (_BYTE)v20 == 18;
    LODWORD(v59) = v21;
    v19 = (__int64 *)sub_BCE1B0(v19, v59);
  }
  v55 = v19;
  if ( *(_BYTE *)v50 == 42 )
  {
    v37 = 311;
    goto LABEL_62;
  }
  if ( *(_BYTE *)v50 != 44 )
    return 0;
  v37 = 338;
LABEL_62:
  if ( v8 < (unsigned int)sub_9AF930(*(_QWORD *)(v50 - 64), a1[11], 0, a1[8], v50, a1[10])
    || v8 < (unsigned int)sub_9AF930(*(_QWORD *)(v50 - 32), a1[11], 0, a1[8], v50, a1[10]) )
  {
    return 0;
  }
  v22 = v50;
  v23 = (__int64)v55;
  v24 = a1[4];
  v63 = 257;
  if ( v55 == *(__int64 **)(*(_QWORD *)(v50 - 64) + 8LL) )
  {
    v25 = *(void **)(v50 - 64);
  }
  else
  {
    v39 = (__int64)v55;
    v43 = *(_QWORD *)(v50 - 64);
    v25 = (void *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v24 + 80) + 120LL))(
                    *(_QWORD *)(v24 + 80),
                    38,
                    v43,
                    v55);
    if ( v25 )
    {
      v22 = v50;
      v23 = (__int64)v55;
      v24 = a1[4];
    }
    else
    {
      v67 = 257;
      v25 = (void *)sub_B51D30(38, v43, v39, (__int64)&v64, 0, 0);
      (*(void (__fastcall **)(_QWORD, void *, void **, _QWORD, _QWORD))(**(_QWORD **)(v24 + 88) + 16LL))(
        *(_QWORD *)(v24 + 88),
        v25,
        &v61,
        *(_QWORD *)(v24 + 56),
        *(_QWORD *)(v24 + 64));
      v32 = *(_QWORD *)v24 + 16LL * *(unsigned int *)(v24 + 8);
      v33 = *(_QWORD *)v24;
      v47 = v32;
      while ( v47 != v33 )
      {
        v34 = *(_QWORD *)(v33 + 8);
        v35 = *(_DWORD *)v33;
        v33 += 16;
        sub_B99FD0((__int64)v25, v35, v34);
      }
      v22 = v50;
      v23 = (__int64)v55;
      v24 = a1[4];
    }
  }
  v63 = 257;
  if ( v23 == *(_QWORD *)(*(_QWORD *)(v22 - 32) + 8LL) )
  {
    v26 = *(_QWORD *)(v22 - 32);
  }
  else
  {
    v40 = v23;
    v44 = *(_QWORD *)(v22 - 32);
    v26 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v24 + 80) + 120LL))(
            *(_QWORD *)(v24 + 80),
            38,
            v44,
            v23);
    if ( !v26 )
    {
      v67 = 257;
      v45 = sub_B51D30(38, v44, v40, (__int64)&v64, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, void **, _QWORD, _QWORD))(**(_QWORD **)(v24 + 88) + 16LL))(
        *(_QWORD *)(v24 + 88),
        v45,
        &v61,
        *(_QWORD *)(v24 + 56),
        *(_QWORD *)(v24 + 64));
      v26 = v45;
      v28 = *(_QWORD *)v24 + 16LL * *(unsigned int *)(v24 + 8);
      v29 = *(_QWORD *)v24;
      for ( i = v28; i != v29; v26 = v46 )
      {
        v30 = *(_QWORD *)(v29 + 8);
        v31 = *(_DWORD *)v29;
        v29 += 16;
        v46 = v26;
        sub_B99FD0(v26, v31, v30);
      }
    }
    v24 = a1[4];
  }
  v62 = v26;
  HIDWORD(v57) = 0;
  v67 = 257;
  v61 = v25;
  v27 = sub_B33D10(v24, v37, (__int64)&v55, 1, (int)&v61, 2, (unsigned int)v57, (__int64)&v64);
  v67 = 257;
  return sub_B51D30(40, v27, v2, (__int64)&v64, 0, 0);
}
