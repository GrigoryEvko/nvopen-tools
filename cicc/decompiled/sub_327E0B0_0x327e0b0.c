// Function: sub_327E0B0
// Address: 0x327e0b0
//
__int64 __fastcall sub_327E0B0(__int64 *a1, __int64 a2)
{
  __int128 v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // r15
  int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rax
  __int64 (__fastcall *v12)(__int64, __int64); // rax
  __int64 v13; // r14
  int v14; // eax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rcx
  int v18; // eax
  bool v19; // dl
  int v20; // eax
  const void *v21; // rsi
  int v22; // esi
  __int64 v23; // rdi
  int v24; // r11d
  __int64 v25; // r14
  __int64 v26; // rax
  int v27; // r9d
  __int128 v28; // rax
  unsigned int *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // r8
  __int64 v32; // rax
  unsigned int v33; // r9d
  int v34; // edx
  __int64 v35; // rax
  unsigned int v36; // r8d
  int v37; // ecx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rsi
  unsigned __int16 *v49; // rax
  unsigned __int16 *v50; // rax
  int v51; // ecx
  int v52; // r9d
  __int64 v53; // rdx
  __int128 v54; // rax
  int v55; // r9d
  __int128 v56; // rax
  __int64 v57; // r14
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rdi
  int v61; // eax
  int v63; // [rsp+0h] [rbp-100h]
  __int64 v64; // [rsp+8h] [rbp-F8h]
  int v65; // [rsp+10h] [rbp-F0h]
  __int64 v66; // [rsp+18h] [rbp-E8h]
  int v67; // [rsp+18h] [rbp-E8h]
  __int64 v68; // [rsp+20h] [rbp-E0h]
  int v69; // [rsp+20h] [rbp-E0h]
  __int128 v70; // [rsp+30h] [rbp-D0h]
  __int64 *v71; // [rsp+40h] [rbp-C0h]
  unsigned int v72; // [rsp+48h] [rbp-B8h]
  unsigned int v73; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v74; // [rsp+48h] [rbp-B8h]
  __int64 v75; // [rsp+50h] [rbp-B0h]
  int v76; // [rsp+58h] [rbp-A8h]
  int v77; // [rsp+5Ch] [rbp-A4h]
  __int64 v78; // [rsp+60h] [rbp-A0h]
  __int64 v79; // [rsp+60h] [rbp-A0h]
  __int128 v80; // [rsp+60h] [rbp-A0h]
  __int128 v81; // [rsp+70h] [rbp-90h]
  __int64 v82; // [rsp+70h] [rbp-90h]
  __int128 v83; // [rsp+70h] [rbp-90h]
  int v84; // [rsp+80h] [rbp-80h]
  __int64 v85; // [rsp+80h] [rbp-80h]
  __int128 v86; // [rsp+80h] [rbp-80h]
  __int128 v87; // [rsp+80h] [rbp-80h]
  __int64 v88; // [rsp+80h] [rbp-80h]
  __int64 v89; // [rsp+90h] [rbp-70h] BYREF
  int v90; // [rsp+98h] [rbp-68h]
  const void *v91; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v92; // [rsp+A8h] [rbp-58h]
  _OWORD v93[5]; // [rsp+B0h] [rbp-50h] BYREF

  if ( (unsigned __int8)sub_33DFCF0(**(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 0) )
  {
    *(_QWORD *)&v4 = 0;
    return v4;
  }
  v5 = *(__int64 **)(a2 + 40);
  v6 = *v5;
  v7 = *((_DWORD *)v5 + 2);
  v8 = *(_QWORD *)(*v5 + 56);
  if ( !v8 )
    goto LABEL_14;
  v9 = *(_QWORD *)(*v5 + 56);
  v10 = 1;
  do
  {
    while ( *(_DWORD *)(v9 + 8) != v7 )
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        goto LABEL_11;
    }
    if ( !v10 )
      goto LABEL_14;
    v11 = *(_QWORD *)(v9 + 32);
    if ( !v11 )
      goto LABEL_12;
    if ( v7 == *(_DWORD *)(v11 + 8) )
      goto LABEL_14;
    v9 = *(_QWORD *)(v11 + 32);
    v10 = 0;
  }
  while ( v9 );
LABEL_11:
  if ( v10 == 1 )
    goto LABEL_14;
LABEL_12:
  v12 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1[1] + 2152LL);
  if ( v12 == sub_302E0E0 )
  {
    if ( *(_QWORD *)(v8 + 32) )
      goto LABEL_14;
    v13 = *v5;
    if ( *(_DWORD *)(v6 + 24) == 213 )
    {
      v15 = *(_QWORD *)(**(_QWORD **)(v6 + 40) + 56LL);
      if ( !v15 || *(_QWORD *)(v15 + 32) )
        goto LABEL_14;
    }
  }
  else
  {
    if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v12)(a1[1], a2, *((unsigned int *)a1 + 6)) )
      goto LABEL_14;
    v5 = *(__int64 **)(a2 + 40);
    v13 = *v5;
    v7 = *((_DWORD *)v5 + 2);
    v8 = *(_QWORD *)(*v5 + 56);
    if ( !v8 )
    {
LABEL_31:
      v16 = *(_DWORD *)(v6 + 24);
      if ( v16 == 56 )
      {
        if ( *(_DWORD *)(a2 + 24) != 190 )
          goto LABEL_14;
      }
      else if ( (unsigned int)(v16 - 186) > 2 )
      {
        goto LABEL_14;
      }
      v17 = *(_QWORD *)(v6 + 40);
      v18 = *(_DWORD *)(*(_QWORD *)v17 + 24LL);
      v19 = v18 == 205 || v18 == 50;
      if ( (unsigned int)(v18 - 190) > 2 )
      {
        if ( !v19 )
          goto LABEL_14;
      }
      else if ( !v19 )
      {
        v20 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v17 + 40LL) + 40LL) + 24LL);
        if ( v20 == 35 || v20 == 11 )
        {
LABEL_37:
          v21 = *(const void **)(a2 + 80);
          v91 = v21;
          if ( v21 )
          {
            sub_B96E90((__int64)&v91, (__int64)v21, 1);
            v17 = *(_QWORD *)(v6 + 40);
          }
          v22 = *(_DWORD *)(a2 + 24);
          v23 = *a1;
          v92 = *(_DWORD *)(a2 + 72);
          v24 = **(unsigned __int16 **)(a2 + 48);
          v25 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
          v26 = *(_QWORD *)(a2 + 40);
          v93[0] = _mm_loadu_si128((const __m128i *)(v17 + 40));
          v84 = v24;
          v93[1] = _mm_loadu_si128((const __m128i *)(v26 + 40));
          *(_QWORD *)&v4 = sub_3402EA0(v23, v22, (unsigned int)&v91, v24, v25, 0, (__int64)v93, 2);
          v81 = v4;
          if ( (_QWORD)v4 )
          {
            *(_QWORD *)&v28 = sub_3406EB0(
                                *a1,
                                *(_DWORD *)(a2 + 24),
                                (unsigned int)&v91,
                                v84,
                                v25,
                                v27,
                                *(_OWORD *)*(_QWORD *)(v6 + 40),
                                *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
            *(_QWORD *)&v4 = sub_3406EB0(*a1, *(_DWORD *)(v6 + 24), (unsigned int)&v91, v84, v25, v81, v28, v81);
          }
          if ( v91 )
          {
            v85 = v4;
            sub_B91220((__int64)&v91, (__int64)v91);
            *(_QWORD *)&v4 = v85;
          }
          return v4;
        }
LABEL_14:
        *(_QWORD *)&v4 = 0;
        return v4;
      }
      v41 = *(_QWORD *)(a2 + 56);
      if ( !v41 || *(_QWORD *)(v41 + 32) )
        goto LABEL_37;
      goto LABEL_14;
    }
  }
  v14 = 1;
  do
  {
    if ( *(_DWORD *)(v8 + 8) == v7 )
    {
      if ( !v14 )
        goto LABEL_31;
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_30;
      if ( v7 == *(_DWORD *)(v8 + 8) )
        goto LABEL_31;
      v14 = 0;
    }
    v8 = *(_QWORD *)(v8 + 32);
  }
  while ( v8 );
  if ( v14 == 1 )
    goto LABEL_31;
LABEL_30:
  v77 = *(_DWORD *)(v13 + 24);
  if ( (unsigned int)(v77 - 186) > 2 )
    goto LABEL_31;
  v70 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 5));
  v75 = *a1;
  v76 = *(_DWORD *)(a2 + 24);
  v82 = *(_QWORD *)(sub_33DFBC0(v70, *((_QWORD *)&v70 + 1), 0, 0) + 96);
  *((_QWORD *)&v86 + 1) = 0;
  v29 = *(unsigned int **)(v13 + 40);
  v71 = (__int64 *)(v82 + 24);
  v30 = *(_QWORD *)v29;
  if ( v76 != *(_DWORD *)(*(_QWORD *)v29 + 24LL) )
    goto LABEL_44;
  v35 = *(_QWORD *)(v30 + 56);
  if ( !v35 )
    goto LABEL_44;
  v36 = v29[2];
  v37 = 1;
  do
  {
    if ( *(_DWORD *)(v35 + 8) == v36 )
    {
      if ( !v37 )
        goto LABEL_44;
      v35 = *(_QWORD *)(v35 + 32);
      if ( !v35 )
        goto LABEL_64;
      if ( v36 == *(_DWORD *)(v35 + 8) )
        goto LABEL_44;
      v37 = 0;
    }
    v35 = *(_QWORD *)(v35 + 32);
  }
  while ( v35 );
  if ( v37 == 1 )
    goto LABEL_44;
LABEL_64:
  v72 = v29[2];
  v78 = *(_QWORD *)v29;
  v38 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v30 + 40) + 48LL), 0, 0);
  if ( v38 )
  {
    v39 = *(_QWORD *)(v78 + 40);
    v40 = *(_QWORD *)(v38 + 96);
    *(_QWORD *)&v86 = *(_QWORD *)v39;
    *((_QWORD *)&v86 + 1) = *(unsigned int *)(v39 + 8);
    if ( *(_DWORD *)(v40 + 32) == *(_DWORD *)(v82 + 32) )
    {
      LOBYTE(v91) = 0;
      v66 = v40 + 24;
      sub_C49AB0((__int64)v93, (__int64)v71, (__int64 *)(v40 + 24), (bool *)&v91);
      if ( (_BYTE)v91 )
      {
        v60 = *(_QWORD *)&v93[0];
        if ( DWORD2(v93[0]) <= 0x40 )
          goto LABEL_66;
      }
      else
      {
        v58 = sub_3263630(v78, v72);
        v69 = DWORD2(v93[0]);
        if ( DWORD2(v93[0]) <= 0x40 )
        {
          if ( v58 <= *(_QWORD *)&v93[0] )
            goto LABEL_66;
LABEL_96:
          v59 = *(_QWORD *)(v13 + 40);
          v47 = *(unsigned int *)(v59 + 48);
          *(_QWORD *)&v83 = *(_QWORD *)(v59 + 40);
          goto LABEL_78;
        }
        v74 = v58;
        v61 = sub_C444A0((__int64)v93);
        v60 = *(_QWORD *)&v93[0];
        if ( (unsigned int)(v69 - v61) <= 0x40 && v74 > **(_QWORD **)&v93[0] )
        {
          j_j___libc_free_0_0(*(unsigned __int64 *)&v93[0]);
          goto LABEL_96;
        }
      }
      if ( v60 )
        j_j___libc_free_0_0(v60);
    }
  }
LABEL_66:
  v29 = *(unsigned int **)(v13 + 40);
LABEL_44:
  v31 = *((_QWORD *)v29 + 5);
  if ( v76 != *(_DWORD *)(v31 + 24) )
    goto LABEL_31;
  v32 = *(_QWORD *)(v31 + 56);
  if ( !v32 )
    goto LABEL_31;
  v33 = v29[12];
  v34 = 1;
  do
  {
    if ( *(_DWORD *)(v32 + 8) == v33 )
    {
      if ( !v34 )
        goto LABEL_31;
      v32 = *(_QWORD *)(v32 + 32);
      if ( !v32 )
        goto LABEL_72;
      if ( v33 == *(_DWORD *)(v32 + 8) )
        goto LABEL_31;
      v34 = 0;
    }
    v32 = *(_QWORD *)(v32 + 32);
  }
  while ( v32 );
  if ( v34 == 1 )
    goto LABEL_31;
LABEL_72:
  v73 = v33;
  v79 = v31;
  v42 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v31 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v31 + 40) + 48LL), 0, 0);
  if ( !v42 )
    goto LABEL_31;
  v43 = *(_QWORD *)(v42 + 96);
  v44 = *(_QWORD *)(v79 + 40);
  *(_QWORD *)&v86 = *(_QWORD *)v44;
  *((_QWORD *)&v86 + 1) = *(unsigned int *)(v44 + 8) | *((_QWORD *)&v86 + 1) & 0xFFFFFFFF00000000LL;
  if ( *(_DWORD *)(v43 + 32) != *(_DWORD *)(v82 + 32) )
    goto LABEL_31;
  LOBYTE(v91) = 0;
  v66 = v43 + 24;
  sub_C49AB0((__int64)v93, (__int64)v71, (__int64 *)(v43 + 24), (bool *)&v91);
  if ( (_BYTE)v91 || (v45 = sub_3263630(v79, v73), !sub_986EE0((__int64)v93, v45)) )
  {
    if ( DWORD2(v93[0]) > 0x40 && *(_QWORD *)&v93[0] )
      j_j___libc_free_0_0(*(unsigned __int64 *)&v93[0]);
    goto LABEL_31;
  }
  if ( DWORD2(v93[0]) > 0x40 && *(_QWORD *)&v93[0] )
    j_j___libc_free_0_0(*(unsigned __int64 *)&v93[0]);
  v46 = *(_QWORD *)(v13 + 40);
  v47 = *(unsigned int *)(v46 + 8);
  *(_QWORD *)&v83 = *(_QWORD *)v46;
LABEL_78:
  v48 = *(_QWORD *)(a2 + 80);
  v89 = v48;
  *((_QWORD *)&v83 + 1) = v47;
  if ( v48 )
    sub_B96E90((__int64)&v89, v48, 1);
  v90 = *(_DWORD *)(a2 + 72);
  v49 = *(unsigned __int16 **)(a2 + 48);
  v68 = *((_QWORD *)v49 + 1);
  v65 = *v49;
  v50 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL));
  v64 = *((_QWORD *)v50 + 1);
  v51 = *v50;
  v92 = *(_DWORD *)(v66 + 8);
  if ( v92 > 0x40 )
  {
    v63 = v51;
    sub_C43780((__int64)&v91, (const void **)v66);
    v51 = v63;
  }
  else
  {
    v91 = *(const void **)v66;
  }
  v67 = v51;
  sub_C45EE0((__int64)&v91, v71);
  DWORD2(v93[0]) = v92;
  v92 = 0;
  *(_QWORD *)&v93[0] = v91;
  *(_QWORD *)&v80 = sub_34007B0(v75, (unsigned int)v93, (unsigned int)&v89, v67, v64, 0, 0);
  *((_QWORD *)&v80 + 1) = v53;
  if ( DWORD2(v93[0]) > 0x40 && *(_QWORD *)&v93[0] )
    j_j___libc_free_0_0(*(unsigned __int64 *)&v93[0]);
  if ( v92 > 0x40 && v91 )
    j_j___libc_free_0_0((unsigned __int64)v91);
  *(_QWORD *)&v54 = sub_3406EB0(v75, v76, (unsigned int)&v89, v65, v68, v52, v86, v80);
  v87 = v54;
  *(_QWORD *)&v56 = sub_3406EB0(v75, v76, (unsigned int)&v89, v65, v68, v55, v83, v70);
  *(_QWORD *)&v4 = sub_3405C90(v75, v77, (unsigned int)&v89, v65, v68, *(_DWORD *)(v13 + 28), v87, v56);
  v57 = v4;
  if ( v89 )
  {
    v88 = v4;
    sub_B91220((__int64)&v89, v89);
    *(_QWORD *)&v4 = v88;
  }
  if ( !v57 )
    goto LABEL_31;
  return v4;
}
