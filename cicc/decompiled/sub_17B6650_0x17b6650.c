// Function: sub_17B6650
// Address: 0x17b6650
//
__int64 __fastcall sub_17B6650(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 *i; // r14
  _QWORD *v11; // rbx
  int v12; // eax
  __int64 v13; // r14
  _QWORD *v14; // rax
  _QWORD *v15; // r15
  _QWORD *v16; // rax
  __int64 v17; // r14
  _QWORD *v18; // rdi
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __m128i v23; // xmm0
  __int64 v24; // rcx
  _QWORD *v25; // r15
  _QWORD *v26; // rax
  __int64 v27; // r14
  _QWORD *v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v34; // rax
  unsigned __int8 *v35; // rsi
  char v36; // al
  _QWORD *v37; // r8
  int v38; // r9d
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // rdi
  __int64 v44; // [rsp+8h] [rbp-278h]
  __int64 v46; // [rsp+10h] [rbp-270h]
  __int64 v47; // [rsp+18h] [rbp-268h]
  _BYTE *v48; // [rsp+28h] [rbp-258h]
  __int64 v49; // [rsp+28h] [rbp-258h]
  _QWORD *v50; // [rsp+28h] [rbp-258h]
  __int64 v51; // [rsp+30h] [rbp-250h] BYREF
  __int64 *v52; // [rsp+38h] [rbp-248h] BYREF
  _BYTE v53[16]; // [rsp+40h] [rbp-240h] BYREF
  __int16 v54; // [rsp+50h] [rbp-230h]
  unsigned __int8 *v55; // [rsp+60h] [rbp-220h] BYREF
  __int64 v56; // [rsp+68h] [rbp-218h]
  __int64 *v57; // [rsp+70h] [rbp-210h]
  __int64 v58; // [rsp+78h] [rbp-208h]
  __int64 v59; // [rsp+80h] [rbp-200h]
  int v60; // [rsp+88h] [rbp-1F8h]
  __m128i v61; // [rsp+90h] [rbp-1F0h] BYREF
  __int64 v62; // [rsp+A0h] [rbp-1E0h]
  unsigned __int8 *v63; // [rsp+B0h] [rbp-1D0h] BYREF
  __int64 v64; // [rsp+B8h] [rbp-1C8h]
  __int64 *v65; // [rsp+C0h] [rbp-1C0h]
  __int64 v66; // [rsp+C8h] [rbp-1B8h]
  __int64 v67; // [rsp+D0h] [rbp-1B0h]
  int v68; // [rsp+D8h] [rbp-1A8h]
  __m128i v69; // [rsp+E0h] [rbp-1A0h]
  __int64 v70; // [rsp+F0h] [rbp-190h]
  _BYTE *v71; // [rsp+100h] [rbp-180h] BYREF
  __int64 v72; // [rsp+108h] [rbp-178h]
  _BYTE v73[64]; // [rsp+110h] [rbp-170h] BYREF
  _BYTE v74[24]; // [rsp+150h] [rbp-130h] BYREF
  __int64 v75[12]; // [rsp+168h] [rbp-118h] BYREF
  _QWORD *v76; // [rsp+1C8h] [rbp-B8h]
  unsigned int v77; // [rsp+1D8h] [rbp-A8h]
  __int64 v78; // [rsp+1E8h] [rbp-98h]
  unsigned __int64 v79; // [rsp+1F0h] [rbp-90h]

  v6 = a1 + 72;
  v7 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v8 = sub_15E0530(a1);
  sub_140CFB0((__int64)v74, v7, a2, v8, 1);
  v9 = *(_QWORD *)(a1 + 80);
  v71 = v73;
  v72 = 0x400000000LL;
  if ( a1 + 72 == v9 )
  {
    i = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    while ( 1 )
    {
      i = *(__int64 **)(v9 + 24);
      if ( i != (__int64 *)(v9 + 16) )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v6 == v9 )
        goto LABEL_7;
      if ( !v9 )
        BUG();
    }
  }
  while ( v9 != v6 )
  {
    if ( !i )
      BUG();
    v49 = i[2];
    v34 = sub_157E9C0(v49);
    v63 = 0;
    v66 = v34;
    v64 = v49;
    v67 = 0;
    v68 = 0;
    v69 = 0u;
    v70 = v7;
    v65 = i;
    if ( i != (__int64 *)(v49 + 40) )
    {
      v35 = (unsigned __int8 *)i[3];
      v55 = v35;
      if ( v35 )
      {
        sub_1623A60((__int64)&v55, (__int64)v35, 2);
        if ( v63 )
          sub_161E7C0((__int64)&v63, (__int64)v63);
        v63 = v55;
        if ( v55 )
          sub_1623210((__int64)&v55, v55, (__int64)&v63);
      }
    }
    v36 = *((_BYTE *)i - 8);
    if ( v36 == 54 )
    {
      v37 = sub_17B54E0((__int64 *)*(i - 6), *(i - 3), v7, (__int64)v74, (__int64 *)&v63, a3, a4, a5, a6);
    }
    else
    {
      switch ( v36 )
      {
        case '7':
          v42 = (__int64 *)*(i - 9);
          v43 = (__int64 *)*(i - 6);
          break;
        case ':':
          v42 = (__int64 *)*(i - 9);
          v43 = (__int64 *)*(i - 12);
          break;
        case ';':
          v42 = (__int64 *)*(i - 6);
          v43 = (__int64 *)*(i - 9);
          break;
        default:
          goto LABEL_67;
      }
      v37 = sub_17B54E0(v43, *v42, v7, (__int64)v74, (__int64 *)&v63, a3, a4, a5, a6);
    }
    if ( v37 )
    {
      v39 = (unsigned int)v72;
      if ( (unsigned int)v72 >= HIDWORD(v72) )
      {
        v50 = v37;
        sub_16CD150((__int64)&v71, v73, 0, 16, (int)v37, v38);
        v39 = (unsigned int)v72;
        v37 = v50;
      }
      v40 = &v71[16 * v39];
      v40[1] = v37;
      *v40 = i - 3;
      LODWORD(v72) = v72 + 1;
    }
LABEL_67:
    if ( v63 )
      sub_161E7C0((__int64)&v63, (__int64)v63);
    for ( i = (__int64 *)i[1]; ; i = *(__int64 **)(v9 + 24) )
    {
      v41 = v9 - 24;
      if ( !v9 )
        v41 = 0;
      if ( i != (__int64 *)(v41 + 40) )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v6 == v9 )
        goto LABEL_7;
      if ( !v9 )
        BUG();
    }
  }
LABEL_7:
  v11 = v71;
  v51 = 0;
  v12 = v72;
  v13 = 16LL * (unsigned int)v72;
  v48 = &v71[v13];
  if ( v71 != &v71[v13] )
  {
    v47 = v7;
    do
    {
      v19 = *v11;
      v7 = *(_QWORD *)(*v11 + 40LL);
      v20 = *v11 + 24LL;
      v21 = sub_157E9C0(v7);
      v55 = 0;
      v58 = v21;
      v59 = 0;
      v60 = 0;
      v61 = 0u;
      v62 = v47;
      v56 = v7;
      v57 = (__int64 *)v20;
      if ( v20 == v7 + 40 )
        goto LABEL_32;
      v22 = *(unsigned __int8 **)(v19 + 48);
      v63 = v22;
      if ( !v22 )
        goto LABEL_32;
      sub_1623A60((__int64)&v63, (__int64)v22, 2);
      if ( v55 )
        sub_161E7C0((__int64)&v55, (__int64)v55);
      v55 = v63;
      if ( v63 )
      {
        sub_1623210((__int64)&v63, v63, (__int64)&v55);
        v63 = v55;
        if ( v55 )
          sub_1623A60((__int64)&v63, (__int64)v55, 2);
      }
      else
      {
LABEL_32:
        v63 = 0;
      }
      v23 = _mm_load_si128(&v61);
      v64 = v56;
      v69 = v23;
      v65 = v57;
      v66 = v58;
      v67 = v59;
      v68 = v60;
      v70 = v62;
      v24 = v11[1];
      v52 = &v51;
      if ( v24 && *(_BYTE *)(v24 + 16) == 13 )
      {
        v14 = *(_QWORD **)(v24 + 24);
        if ( *(_DWORD *)(v24 + 32) > 0x40u )
          v14 = (_QWORD *)*v14;
        if ( v14 )
        {
          if ( !v65 )
            BUG();
          v15 = (_QWORD *)v65[2];
          v54 = 257;
          sub_157FBF0(v15, v65, (__int64)v53);
          v16 = (_QWORD *)sub_157EBA0((__int64)v15);
          sub_15F20C0(v16);
          v17 = sub_17B61B0(&v52, (__int64 *)&v63);
          v18 = sub_1648A60(56, 1u);
          if ( v18 )
            sub_15F8590((__int64)v18, v17, (__int64)v15);
        }
      }
      else
      {
        if ( !v65 )
          BUG();
        v25 = (_QWORD *)v65[2];
        v44 = v24;
        v54 = 257;
        v46 = sub_157FBF0(v25, v65, (__int64)v53);
        v26 = (_QWORD *)sub_157EBA0((__int64)v25);
        sub_15F20C0(v26);
        v27 = sub_17B61B0(&v52, (__int64 *)&v63);
        v28 = sub_1648A60(56, 3u);
        if ( v28 )
          sub_15F8650((__int64)v28, v27, v46, v44, (__int64)v25);
      }
      if ( v63 )
        sub_161E7C0((__int64)&v63, (__int64)v63);
      if ( v55 )
        sub_161E7C0((__int64)&v55, (__int64)v55);
      v11 += 2;
    }
    while ( v48 != (_BYTE *)v11 );
    v48 = v71;
    v12 = v72;
  }
  LOBYTE(v7) = v12 != 0;
  if ( v48 != v73 )
    _libc_free((unsigned __int64)v48);
  if ( v79 != v78 )
    _libc_free(v79);
  if ( v77 )
  {
    v29 = v76;
    v30 = &v76[7 * v77];
    do
    {
      if ( *v29 != -16 && *v29 != -8 )
      {
        v31 = v29[6];
        if ( v31 != -8 && v31 != 0 && v31 != -16 )
          sub_1649B30(v29 + 4);
        v32 = v29[3];
        if ( v32 != 0 && v32 != -8 && v32 != -16 )
          sub_1649B30(v29 + 1);
      }
      v29 += 7;
    }
    while ( v30 != v29 );
  }
  j___libc_free_0(v76);
  if ( v75[0] )
    sub_161E7C0((__int64)v75, v75[0]);
  return (unsigned int)v7;
}
