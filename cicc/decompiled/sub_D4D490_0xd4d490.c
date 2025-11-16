// Function: sub_D4D490
// Address: 0xd4d490
//
__int64 __fastcall sub_D4D490(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // r13
  __int32 v7; // eax
  _QWORD *v8; // r12
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  _BYTE *v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // rsi
  __int64 result; // rax
  __int64 v37; // rcx
  _BYTE *v38; // rdx
  char *v39; // r15
  __int64 v40; // rdi
  int v41; // esi
  __int64 v42; // r8
  int v43; // esi
  __int64 v44; // rcx
  __int64 *v45; // rdx
  __int64 v46; // r9
  __int64 v47; // r13
  _BYTE *v48; // rsi
  char *v49; // rsi
  __int64 *v50; // rax
  __int64 *v51; // rdx
  __int64 *v52; // rax
  __int64 *v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rsi
  __int64 *v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rsi
  int v60; // edx
  int v61; // r10d
  char *v62[54]; // [rsp+50h] [rbp-A50h] BYREF
  char *v63; // [rsp+200h] [rbp-8A0h] BYREF
  _BYTE *v64; // [rsp+208h] [rbp-898h]
  __int64 v65; // [rsp+210h] [rbp-890h]
  int v66; // [rsp+218h] [rbp-888h]
  char v67; // [rsp+21Ch] [rbp-884h]
  _BYTE v68[64]; // [rsp+220h] [rbp-880h] BYREF
  _BYTE *v69; // [rsp+260h] [rbp-840h] BYREF
  __int64 v70; // [rsp+268h] [rbp-838h]
  _BYTE v71[320]; // [rsp+270h] [rbp-830h] BYREF
  __int64 v72; // [rsp+3B0h] [rbp-6F0h] BYREF
  __int64 v73; // [rsp+3B8h] [rbp-6E8h]
  char v74; // [rsp+3CCh] [rbp-6D4h]
  _BYTE v75[64]; // [rsp+3D0h] [rbp-6D0h] BYREF
  _BYTE *v76; // [rsp+410h] [rbp-690h] BYREF
  __int64 v77; // [rsp+418h] [rbp-688h]
  _BYTE v78[320]; // [rsp+420h] [rbp-680h] BYREF
  __m128i v79; // [rsp+560h] [rbp-540h] BYREF
  char v80; // [rsp+57Ch] [rbp-524h]
  _BYTE v81[64]; // [rsp+580h] [rbp-520h] BYREF
  _BYTE *v82; // [rsp+5C0h] [rbp-4E0h] BYREF
  __int64 v83; // [rsp+5C8h] [rbp-4D8h]
  _BYTE v84[320]; // [rsp+5D0h] [rbp-4D0h] BYREF
  __m128i v85; // [rsp+710h] [rbp-390h] BYREF
  char v86; // [rsp+72Ch] [rbp-374h]
  char v87[64]; // [rsp+730h] [rbp-370h] BYREF
  _BYTE *v88; // [rsp+770h] [rbp-330h] BYREF
  __int64 v89; // [rsp+778h] [rbp-328h]
  _BYTE v90[320]; // [rsp+780h] [rbp-320h] BYREF
  char v91[8]; // [rsp+8C0h] [rbp-1E0h] BYREF
  __int64 v92; // [rsp+8C8h] [rbp-1D8h]
  char v93; // [rsp+8DCh] [rbp-1C4h]
  _BYTE v94[64]; // [rsp+8E0h] [rbp-1C0h] BYREF
  _BYTE *v95; // [rsp+920h] [rbp-180h] BYREF
  __int64 v96; // [rsp+928h] [rbp-178h]
  _BYTE v97[368]; // [rsp+930h] [rbp-170h] BYREF

  memset(v62, 0, sizeof(v62));
  v72 = a2;
  v62[12] = (char *)&v62[14];
  v64 = v68;
  v69 = v71;
  v62[1] = (char *)&v62[4];
  LODWORD(v62[2]) = 8;
  BYTE4(v62[3]) = 1;
  HIDWORD(v62[13]) = 8;
  v63 = 0;
  v65 = 8;
  v66 = 0;
  v67 = 1;
  v70 = 0x800000000LL;
  sub_AE6EC0((__int64)&v63, a2);
  v3 = v72;
  v6 = sub_986580(v72);
  v7 = 0;
  if ( v6 )
    v7 = sub_B46E30(v6);
  v85.m128i_i64[0] = v6;
  v8 = (_QWORD *)(v3 + 48);
  v85.m128i_i32[2] = v7;
  v9 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v9 == v8 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = v9 - 24;
    if ( (unsigned int)(v10 - 30) >= 0xB )
      v11 = 0;
  }
  v79.m128i_i64[0] = v11;
  v79.m128i_i32[2] = 0;
  sub_D46560((__int64)&v69, &v72, &v79, &v85, v4, v5);
  sub_D4D230((__int64)&v63);
  sub_C8CF70((__int64)&v79, v81, 8, (__int64)&v62[4], (__int64)v62);
  v82 = v84;
  v83 = 0x800000000LL;
  if ( LODWORD(v62[13]) )
    sub_D4C550((__int64)&v82, (__int64)&v62[12], v12, v13, v14, v15);
  sub_C8CF70((__int64)&v72, v75, 8, (__int64)v68, (__int64)&v63);
  v76 = v78;
  v77 = 0x800000000LL;
  if ( (_DWORD)v70 )
    sub_D4C550((__int64)&v76, (__int64)&v69, v16, v17, v18, v19);
  sub_C8CF70((__int64)&v85, v87, 8, (__int64)v75, (__int64)&v72);
  v88 = v90;
  v89 = 0x800000000LL;
  if ( (_DWORD)v77 )
    sub_D4C550((__int64)&v88, (__int64)&v76, v20, (unsigned int)v77, v21, v22);
  v23 = v94;
  sub_C8CF70((__int64)v91, v94, 8, (__int64)v81, (__int64)&v79);
  v95 = v97;
  v96 = 0x800000000LL;
  if ( (_DWORD)v83 )
  {
    v23 = &v82;
    sub_D4C550((__int64)&v95, (__int64)&v82, (unsigned int)v83, v24, v25, v26);
  }
  if ( v76 != v78 )
    _libc_free(v76, v23);
  if ( !v74 )
    _libc_free(v73, v23);
  if ( v82 != v84 )
    _libc_free(v82, v23);
  if ( !v80 )
    _libc_free(v79.m128i_i64[1], v23);
  if ( v69 != v71 )
    _libc_free(v69, v23);
  if ( !v67 )
    _libc_free(v64, v23);
  if ( (char **)v62[12] != &v62[14] )
    _libc_free(v62[12], v23);
  if ( !BYTE4(v62[3]) )
    _libc_free(v62[1], v23);
  sub_C8CD80((__int64)&v72, (__int64)v75, (__int64)&v85, v24, v25, v26);
  v76 = v78;
  v77 = 0x800000000LL;
  if ( (_DWORD)v89 )
    sub_D4C3E0((__int64)&v76, (__int64 *)&v88, v27, v28, v29, v30);
  sub_C8CD80((__int64)&v79, (__int64)v81, (__int64)v91, v28, v29, v30);
  v35 = (unsigned int)v96;
  v82 = v84;
  v83 = 0x800000000LL;
  if ( (_DWORD)v96 )
  {
    sub_D4C3E0((__int64)&v82, (__int64 *)&v95, v31, v32, v33, v34);
    v35 = (unsigned int)v83;
  }
LABEL_35:
  result = (unsigned int)v77;
  while ( 1 )
  {
    v37 = 40LL * (unsigned int)result;
    if ( (unsigned int)result != v35 )
      goto LABEL_40;
    if ( v76 == &v76[v37] )
      break;
    v35 = (unsigned __int64)v82;
    v38 = v76;
    while ( *((_QWORD *)v38 + 4) == *(_QWORD *)(v35 + 32)
         && *((_DWORD *)v38 + 6) == *(_DWORD *)(v35 + 24)
         && *((_DWORD *)v38 + 2) == *(_DWORD *)(v35 + 8) )
    {
      v38 += 40;
      v35 += 40LL;
      if ( &v76[v37] == v38 )
        goto LABEL_65;
    }
LABEL_40:
    v39 = *(char **)&v76[v37 - 8];
    v40 = *a1;
    v41 = *(_DWORD *)(*a1 + 24);
    v42 = *(_QWORD *)(*a1 + 8);
    if ( !v41 )
      goto LABEL_55;
    v43 = v41 - 1;
    v44 = v43 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v45 = (__int64 *)(v42 + 16 * v44);
    v46 = *v45;
    if ( v39 == (char *)*v45 )
    {
LABEL_42:
      v47 = v45[1];
      v62[0] = (char *)v47;
      if ( !v47 )
        goto LABEL_55;
      if ( v39 == **(char ***)(v47 + 32) )
      {
        if ( *(_QWORD *)v47 )
        {
          sub_D4C980(*(_QWORD *)v47 + 8LL, v62);
        }
        else
        {
          v63 = (char *)v47;
          sub_D4C980(v40 + 32, &v63);
        }
        v44 = (__int64)v62[0];
        v51 = (__int64 *)*((_QWORD *)v62[0] + 5);
        v52 = (__int64 *)(*((_QWORD *)v62[0] + 4) + 8LL);
        if ( v51 != v52 )
        {
          v53 = v51 - 1;
          if ( v52 < v53 )
          {
            do
            {
              v54 = *v52;
              v55 = *v53;
              ++v52;
              --v53;
              *(v52 - 1) = v55;
              v53[1] = v54;
            }
            while ( v52 < v53 );
            v44 = (__int64)v62[0];
          }
        }
        v56 = *(__int64 **)(v44 + 16);
        v45 = *(__int64 **)(v44 + 8);
        if ( v56 != v45 )
        {
          v57 = v56 - 1;
          if ( v45 < v57 )
          {
            do
            {
              v58 = *v45;
              v59 = *v57;
              ++v45;
              --v57;
              *(v45 - 1) = v59;
              v57[1] = v58;
            }
            while ( v45 < v57 );
            v44 = (__int64)v62[0];
          }
        }
        v47 = *(_QWORD *)v44;
        v62[0] = (char *)v47;
        if ( !v47 )
        {
LABEL_54:
          LODWORD(result) = v77;
          goto LABEL_55;
        }
      }
      while ( 2 )
      {
        v63 = v39;
        v48 = *(_BYTE **)(v47 + 40);
        if ( v48 == *(_BYTE **)(v47 + 48) )
        {
          sub_9319A0(v47 + 32, v48, &v63);
          v49 = v63;
        }
        else
        {
          if ( v48 )
          {
            *(_QWORD *)v48 = v39;
            v48 = *(_BYTE **)(v47 + 40);
          }
          *(_QWORD *)(v47 + 40) = v48 + 8;
          v49 = v39;
        }
        if ( *(_BYTE *)(v47 + 84) )
        {
          v50 = *(__int64 **)(v47 + 64);
          v44 = *(unsigned int *)(v47 + 76);
          v45 = &v50[v44];
          if ( v50 != v45 )
          {
            while ( v49 != (char *)*v50 )
            {
              if ( v45 == ++v50 )
                goto LABEL_58;
            }
LABEL_53:
            v47 = *(_QWORD *)v62[0];
            v62[0] = (char *)v47;
            if ( !v47 )
              goto LABEL_54;
            continue;
          }
LABEL_58:
          if ( (unsigned int)v44 < *(_DWORD *)(v47 + 72) )
          {
            v44 = (unsigned int)(v44 + 1);
            *(_DWORD *)(v47 + 76) = v44;
            *v45 = (__int64)v49;
            ++*(_QWORD *)(v47 + 56);
            goto LABEL_53;
          }
        }
        break;
      }
      sub_C8CC70(v47 + 56, (__int64)v49, (__int64)v45, v44, v42, v46);
      goto LABEL_53;
    }
    v60 = 1;
    while ( v46 != -4096 )
    {
      v61 = v60 + 1;
      v44 = v43 & (unsigned int)(v44 + v60);
      v45 = (__int64 *)(v42 + 16LL * (unsigned int)v44);
      v46 = *v45;
      if ( v39 == (char *)*v45 )
        goto LABEL_42;
      v60 = v61;
    }
LABEL_55:
    result = (unsigned int)(result - 1);
    LODWORD(v77) = result;
    if ( (_DWORD)result )
    {
      sub_D4D230((__int64)&v72);
      v35 = (unsigned int)v83;
      goto LABEL_35;
    }
    v35 = (unsigned int)v83;
  }
LABEL_65:
  if ( v82 != v84 )
    result = _libc_free(v82, v35);
  if ( !v80 )
    result = _libc_free(v79.m128i_i64[1], v35);
  if ( v76 != v78 )
    result = _libc_free(v76, v35);
  if ( !v74 )
    result = _libc_free(v73, v35);
  if ( v95 != v97 )
    result = _libc_free(v95, v35);
  if ( !v93 )
    result = _libc_free(v92, v35);
  if ( v88 != v90 )
    result = _libc_free(v88, v35);
  if ( !v86 )
    return _libc_free(v85.m128i_i64[1], v35);
  return result;
}
