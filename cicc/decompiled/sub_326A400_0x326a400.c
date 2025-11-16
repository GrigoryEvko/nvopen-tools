// Function: sub_326A400
// Address: 0x326a400
//
__int64 __fastcall sub_326A400(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, signed int *a5, __int64 a6)
{
  unsigned __int16 *v6; // rax
  __int64 v7; // r14
  int v11; // ecx
  signed int v12; // r14d
  signed int *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rdi
  signed int *v16; // rax
  signed int *v17; // rdx
  unsigned __int16 v18; // r15
  __int64 result; // rax
  __int64 v20; // rax
  _BYTE *v21; // rdi
  _DWORD *v22; // rdx
  _DWORD *v23; // rax
  int *v24; // rsi
  int v25; // r10d
  int v26; // r9d
  __int64 v27; // r11
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // r8
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rdx
  int v34; // ecx
  _DWORD *v35; // rax
  _DWORD *v36; // r15
  int v37; // edx
  __int64 i; // rax
  int v39; // edx
  __int64 v40; // rbx
  __int64 *v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rsi
  __int64 v44; // rdx
  int v45; // r9d
  __int64 v46; // r10
  __int64 v47; // r11
  __int64 *v48; // r13
  __int128 *v49; // r14
  __int64 *v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rsi
  __int128 v53; // [rsp-10h] [rbp-F0h]
  __int64 v54; // [rsp+10h] [rbp-D0h]
  __int64 v55; // [rsp+10h] [rbp-D0h]
  __int64 v56; // [rsp+10h] [rbp-D0h]
  __int64 v57; // [rsp+18h] [rbp-C8h]
  int v58; // [rsp+24h] [rbp-BCh]
  int v59; // [rsp+28h] [rbp-B8h]
  __int64 v60; // [rsp+28h] [rbp-B8h]
  __int64 v61; // [rsp+28h] [rbp-B8h]
  __int128 v62; // [rsp+30h] [rbp-B0h]
  __int64 v63; // [rsp+30h] [rbp-B0h]
  __int64 v64; // [rsp+30h] [rbp-B0h]
  __int64 v65; // [rsp+30h] [rbp-B0h]
  __int16 v66; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v67; // [rsp+48h] [rbp-98h]
  __int64 v68; // [rsp+50h] [rbp-90h] BYREF
  int v69; // [rsp+58h] [rbp-88h]
  __int64 v70; // [rsp+60h] [rbp-80h] BYREF
  int v71; // [rsp+68h] [rbp-78h]
  _BYTE *v72; // [rsp+70h] [rbp-70h] BYREF
  __int64 v73; // [rsp+78h] [rbp-68h]
  _BYTE v74[96]; // [rsp+80h] [rbp-60h] BYREF

  *((_QWORD *)&v62 + 1) = a3;
  *(_QWORD *)&v62 = a2;
  v6 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a4 + 40) + 48LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a4 + 40) + 8LL));
  v7 = *v6;
  v67 = *((_QWORD *)v6 + 1);
  LODWORD(v6) = *(_DWORD *)(a4 + 64);
  v66 = v7;
  v59 = (int)v6;
  if ( !(_WORD)v7 )
  {
    if ( sub_3007100((__int64)&v66) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    return 0;
  }
  if ( (unsigned __int16)(v7 - 176) <= 0x34u )
  {
    v54 = a6;
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    a6 = v54;
  }
  v11 = (unsigned __int16)v7;
  if ( !*(_QWORD *)((*a1)[1] + 8 * v7 + 112) )
    return 0;
  v12 = *(_DWORD *)a1[1];
  v13 = &a5[a6];
  v14 = (4 * a6) >> 4;
  v15 = (4 * a6) >> 2;
  if ( v14 > 0 )
  {
    v16 = a5;
    v17 = &a5[4 * v14];
    while ( *v16 < v12 )
    {
      if ( v16[1] >= v12 )
      {
        ++v16;
        goto LABEL_12;
      }
      if ( v16[2] >= v12 )
      {
        v16 += 2;
        goto LABEL_12;
      }
      if ( v16[3] >= v12 )
      {
        v16 += 3;
        goto LABEL_12;
      }
      v16 += 4;
      if ( v17 == v16 )
      {
        v15 = v13 - v16;
        goto LABEL_61;
      }
    }
    goto LABEL_12;
  }
  v16 = a5;
LABEL_61:
  if ( v15 == 2 )
    goto LABEL_68;
  if ( v15 == 3 )
  {
    if ( *v16 >= v12 )
      goto LABEL_12;
    ++v16;
LABEL_68:
    if ( *v16 >= v12 )
      goto LABEL_12;
    ++v16;
    goto LABEL_64;
  }
  if ( v15 != 1 )
    return 0;
LABEL_64:
  if ( *v16 < v12 )
    return 0;
LABEL_12:
  v18 = word_4456340[v11 - 1];
  if ( v13 == v16 )
    return 0;
  v20 = (unsigned int)v12;
  v21 = v74;
  v72 = v74;
  v73 = 0xC00000000LL;
  if ( v12 )
  {
    v22 = v74;
    if ( (unsigned int)v12 > 0xCuLL )
    {
      sub_C8D5F0((__int64)&v72, v74, (unsigned int)v12, 4u, (__int64)&v72, a6);
      v21 = v72;
      v20 = (unsigned int)v12;
      v22 = &v72[4 * (unsigned int)v73];
    }
    v23 = &v21[4 * v20];
    if ( v23 != v22 )
    {
      do
      {
        if ( v22 )
          *v22 = 0;
        ++v22;
      }
      while ( v23 != v22 );
      v21 = v72;
    }
    LODWORD(v73) = v12;
  }
  if ( v59 )
  {
    v24 = (int *)a1[1];
    v25 = v18;
    v26 = 0;
    v55 = a4;
    v27 = 4LL * v18;
    v28 = v18;
    v58 = 0;
    while ( !*v24 )
    {
LABEL_44:
      ++v58;
      v26 += v25;
      if ( v59 == v58 )
        goto LABEL_45;
    }
    v29 = 0;
    v30 = 0;
    while ( 1 )
    {
      if ( 4LL * (unsigned int)v73 )
      {
        v31 = 0;
        v32 = (4 * (unsigned __int64)(unsigned int)v73 - 4) >> 2;
        do
        {
          v33 = v31;
          *(_DWORD *)&v21[4 * v31] = v31;
          ++v31;
        }
        while ( v32 != v33 );
        v21 = v72;
        v24 = (int *)a1[1];
      }
      v34 = *v24;
      v35 = &v21[4 * v29];
      v36 = &v21[4 * v29 + v27];
      v37 = *v24 + v26;
      if ( v36 != v35 )
      {
        do
          *v35++ = v37++;
        while ( v36 != v35 );
        v24 = (int *)a1[1];
        v34 = *v24;
      }
      if ( !v34 )
        break;
      v21 = v72;
      for ( i = 0; ; ++i )
      {
        v39 = a5[i];
        if ( *(_DWORD *)&v72[4 * i] != v39 && v39 >= 0 )
          break;
        if ( v34 - 1 == i )
          goto LABEL_46;
      }
      v30 = (unsigned int)(v25 + v30);
      v29 += v28;
      if ( (_DWORD)v30 == v34 )
        goto LABEL_44;
    }
LABEL_46:
    v40 = **a1;
    v41 = a1[2];
    v42 = *v41;
    v43 = *(_QWORD *)(*v41 + 80);
    v70 = v43;
    if ( v43 )
    {
      v60 = v42;
      sub_B96E90((__int64)&v70, v43, 1);
      v42 = v60;
    }
    v71 = *(_DWORD *)(v42 + 72);
    v46 = sub_3400EE0(v40, v29, &v70, 0, v30);
    v47 = v44;
    v48 = a1[3];
    v49 = (__int128 *)(*(_QWORD *)(v55 + 40) + 40LL * v58);
    v50 = a1[2];
    v51 = *v50;
    v52 = *(_QWORD *)(*v50 + 80);
    v68 = v52;
    if ( v52 )
    {
      v56 = v46;
      v57 = v44;
      v61 = v51;
      sub_B96E90((__int64)&v68, v52, 1);
      v46 = v56;
      v47 = v57;
      v51 = v61;
    }
    v69 = *(_DWORD *)(v51 + 72);
    *((_QWORD *)&v53 + 1) = v47;
    *(_QWORD *)&v53 = v46;
    result = sub_340F900(v40, 160, (unsigned int)&v68, *(_DWORD *)v48, v48[1], v45, v62, *v49, v53);
    if ( v68 )
    {
      v63 = result;
      sub_B91220((__int64)&v68, v68);
      result = v63;
    }
    if ( v70 )
    {
      v64 = result;
      sub_B91220((__int64)&v70, v70);
      result = v64;
    }
    v21 = v72;
  }
  else
  {
LABEL_45:
    result = 0;
  }
  if ( v21 != v74 )
  {
    v65 = result;
    _libc_free((unsigned __int64)v21);
    return v65;
  }
  return result;
}
