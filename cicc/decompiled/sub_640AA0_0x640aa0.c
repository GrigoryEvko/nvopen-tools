// Function: sub_640AA0
// Address: 0x640aa0
//
__int64 __fastcall sub_640AA0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // rax
  _QWORD *v4; // r12
  __int64 v5; // r15
  _QWORD *v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 **v9; // rax
  __int64 v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  char v15; // r9
  char v16; // r10
  _QWORD *v17; // r8
  __int64 **v18; // rsi
  __int64 *v19; // rax
  __int64 *v20; // rcx
  int v21; // eax
  __int64 *v22; // r13
  const __m128i *v23; // r14
  const __m128i *v24; // r12
  __m128i *v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rax
  __int32 v28; // esi
  __int64 v29; // r15
  __int64 v30; // r15
  __m128i *v31; // rdx
  __int64 v32; // r15
  __m128i *v33; // rdx
  __int64 *v34; // r12
  const __m128i *v35; // rcx
  _QWORD *v36; // r13
  const __m128i *v37; // r12
  const __m128i *v38; // r14
  __m128i *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rax
  __int32 v43; // edx
  __int64 v44; // r15
  __int64 v45; // rdx
  const __m128i *v46; // r15
  __int64 v47; // rdi
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rcx
  char v52; // al
  __int64 v53; // rsi
  __int64 i; // rax
  _QWORD *v55; // rdx
  __int64 j; // rax
  __int64 v57; // rax
  unsigned int v58; // edx
  __int64 v59; // r12
  __int64 v61; // rax
  __int64 v62; // r9
  __int64 v63; // rax
  _QWORD *v64; // rcx
  _QWORD *v65; // rsi
  __int32 v66; // [rsp+8h] [rbp-D8h]
  __int32 v67; // [rsp+8h] [rbp-D8h]
  const __m128i *v68; // [rsp+8h] [rbp-D8h]
  __int64 v69; // [rsp+10h] [rbp-D0h]
  __int64 v70; // [rsp+10h] [rbp-D0h]
  __int64 v71; // [rsp+10h] [rbp-D0h]
  __int64 v72; // [rsp+10h] [rbp-D0h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  __int64 v74; // [rsp+18h] [rbp-C8h]
  __int64 v75; // [rsp+18h] [rbp-C8h]
  _QWORD *v76; // [rsp+20h] [rbp-C0h]
  __int64 v77; // [rsp+20h] [rbp-C0h]
  __int64 v78; // [rsp+20h] [rbp-C0h]
  int v79; // [rsp+20h] [rbp-C0h]
  int v80; // [rsp+20h] [rbp-C0h]
  __int64 v82; // [rsp+30h] [rbp-B0h]
  const __m128i *v83; // [rsp+30h] [rbp-B0h]
  __int64 v84; // [rsp+38h] [rbp-A8h]
  __int64 v85; // [rsp+48h] [rbp-98h] BYREF
  const __m128i *v86; // [rsp+50h] [rbp-90h] BYREF
  __int64 v87; // [rsp+58h] [rbp-88h]
  __int64 v88; // [rsp+60h] [rbp-80h]
  const __m128i *v89; // [rsp+70h] [rbp-70h] BYREF
  __int64 v90; // [rsp+78h] [rbp-68h]
  __int64 v91; // [rsp+80h] [rbp-60h]
  const __m128i *v92; // [rsp+90h] [rbp-50h] BYREF
  __int64 v93; // [rsp+98h] [rbp-48h]
  __int64 v94; // [rsp+A0h] [rbp-40h]

  v85 = 0;
  v88 = 0;
  v1 = sub_823970(0);
  v87 = 0;
  v86 = (const __m128i *)sub_823970(0);
  v91 = 0;
  v90 = 0;
  v89 = (const __m128i *)sub_823970(0);
  v94 = 0;
  v93 = 0;
  v92 = (const __m128i *)sub_823970(0);
  v84 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v76 = *(_QWORD **)(v84 + 168);
  if ( (*(_BYTE *)(a1 + 194) & 0x40) != 0 )
  {
    v2 = a1;
    do
      v2 = *(_QWORD *)(v2 + 232);
    while ( (*(_BYTE *)(v2 + 194) & 0x40) != 0 );
  }
  else
  {
    v2 = a1;
  }
  v3 = *(_QWORD *)(v84 + 168);
  v4 = (_QWORD *)*v76;
  if ( !*v76 )
  {
    v73 = 0;
    goto LABEL_33;
  }
  v82 = v2;
  v5 = 0;
  v6 = (_QWORD *)v1;
  v7 = 0;
  while ( 1 )
  {
    v8 = 8 * v5;
    v9 = *(__int64 ***)(*(_QWORD *)(v3 + 152) + 176LL);
    if ( v9 )
    {
      while ( 1 )
      {
        if ( ((_BYTE)v9[5] & 4) != 0 )
        {
          v10 = v4[5];
          if ( v9[6] == (__int64 *)v10 )
            break;
        }
        v9 = (__int64 **)*v9;
        if ( !v9 )
          goto LABEL_10;
      }
      if ( (unsigned int)sub_62F640(v10, v82) )
        break;
    }
LABEL_10:
    v4 = (_QWORD *)*v4;
    if ( !v4 )
      goto LABEL_17;
LABEL_11:
    v3 = *(_QWORD *)(v84 + 168);
  }
  if ( v7 == v5 )
  {
    if ( v7 <= 1 )
    {
      v62 = 16;
      v5 = 2;
    }
    else
    {
      v5 = v7 + (v7 >> 1) + 1;
      v62 = 8 * v5;
    }
    v74 = v62;
    v63 = sub_823970(v62);
    if ( v7 > 0 )
    {
      v64 = (_QWORD *)v63;
      v65 = v6;
      do
      {
        if ( v64 )
          *v64 = *v65;
        ++v64;
        ++v65;
      }
      while ( (_QWORD *)(v63 + 8 * v7) != v64 );
    }
    v72 = v74;
    v75 = v63;
    sub_823A00(v6, v8);
    v8 = v72;
    v6 = (_QWORD *)v75;
  }
  v11 = &v6[v7];
  if ( v11 )
    *v11 = v4;
  v4 = (_QWORD *)*v4;
  ++v7;
  if ( v4 )
    goto LABEL_11;
LABEL_17:
  v73 = v8;
  v1 = (__int64)v6;
  v12 = v7;
  v2 = v82;
  v13 = (_QWORD *)*v76;
  if ( *v76 )
  {
    v14 = (_QWORD *)(v1 + 8 * v12);
    while ( 1 )
    {
      v15 = *((_BYTE *)v13 + 96);
      v16 = v15 & 2;
      if ( v14 != (_QWORD *)v1 )
        break;
LABEL_30:
      v21 = 0;
      if ( v16 )
        goto LABEL_43;
LABEL_31:
      if ( (v15 & 1) != 0 )
      {
        v32 = v91;
        if ( v91 == v90 )
        {
          v80 = v21;
          sub_640940(&v89);
          v21 = v80;
        }
        v33 = (__m128i *)&v89[v32];
        if ( v33 )
        {
          v33->m128i_i64[0] = (__int64)v13;
          v33->m128i_i32[2] = v21;
        }
        v91 = v32 + 1;
      }
LABEL_32:
      v13 = (_QWORD *)*v13;
      if ( !v13 )
        goto LABEL_33;
    }
    v17 = (_QWORD *)v1;
    while ( (_QWORD *)*v17 != v13 )
    {
      if ( v16 )
      {
        v18 = (__int64 **)v13[14];
        if ( v18 )
        {
          while ( 1 )
          {
            v19 = v18[1];
            v20 = (__int64 *)*v18[2];
            if ( v20 != v19 )
              break;
LABEL_40:
            v18 = (__int64 **)*v18;
            if ( !v18 )
              goto LABEL_29;
          }
          while ( *v17 != v19[2] )
          {
            v19 = (__int64 *)*v19;
            if ( v20 == v19 )
              goto LABEL_40;
          }
          if ( (unsigned int)sub_62F640(v13[5], v82) )
          {
            v21 = 1;
            goto LABEL_43;
          }
        }
      }
LABEL_29:
      if ( v14 == ++v17 )
        goto LABEL_30;
    }
    v21 = 1;
    if ( !v16 )
      goto LABEL_31;
LABEL_43:
    v30 = v88;
    if ( v88 == v87 )
    {
      v79 = v21;
      sub_640940(&v86);
      v21 = v79;
    }
    v31 = (__m128i *)&v86[v30];
    if ( v31 )
    {
      v31->m128i_i64[0] = (__int64)v13;
      v31->m128i_i32[2] = v21;
    }
    v88 = v30 + 1;
    goto LABEL_32;
  }
LABEL_33:
  if ( &v86[v88] == v86 )
  {
    v34 = &v85;
  }
  else
  {
    v77 = v1;
    v69 = v2;
    v22 = &v85;
    v23 = v86;
    v24 = &v86[v88];
    do
    {
      v26 = sub_726BB0(0);
      v27 = v23->m128i_i64[0];
      *(_BYTE *)(v26 + 9) |= 1u;
      *(_QWORD *)(v26 + 16) = v27;
      *v22 = v26;
      v22 = (__int64 *)v26;
      v28 = v23->m128i_i32[2];
      v29 = v94;
      if ( v94 == v93 )
      {
        v66 = v23->m128i_i32[2];
        sub_6409F0(&v92);
        v28 = v66;
      }
      v25 = (__m128i *)&v92[v29];
      if ( v25 )
      {
        v25->m128i_i64[0] = v26;
        v25->m128i_i32[2] = v28;
      }
      ++v23;
      v94 = v29 + 1;
    }
    while ( v24 != v23 );
    v34 = (__int64 *)v26;
    v2 = v69;
    v1 = v77;
  }
  v35 = v89;
  if ( &v89[v91] == v89 )
  {
    v40 = v94;
  }
  else
  {
    v78 = v1;
    v36 = v34;
    v37 = &v89[v91];
    v70 = v2;
    v38 = v89;
    do
    {
      v41 = sub_726BB0(1);
      v42 = v38->m128i_i64[0];
      *(_BYTE *)(v41 + 9) |= 1u;
      *(_QWORD *)(v41 + 16) = v42;
      *v36 = v41;
      v36 = (_QWORD *)v41;
      v43 = v38->m128i_i32[2];
      v44 = v94;
      if ( v94 == v93 )
      {
        v67 = v38->m128i_i32[2];
        sub_6409F0(&v92);
        v43 = v67;
      }
      v39 = (__m128i *)&v92[v44];
      if ( v39 )
      {
        v39->m128i_i64[0] = v41;
        v39->m128i_i32[2] = v43;
      }
      v40 = v44 + 1;
      ++v38;
      v94 = v44 + 1;
    }
    while ( v37 != v38 );
    v34 = (__int64 *)v41;
    v2 = v70;
    v1 = v78;
  }
  v45 = v40;
  v83 = &v92[v45];
  if ( &v92[v45] != v92 )
  {
    v46 = v92;
    v71 = v1;
    while ( 1 )
    {
      v48 = v46->m128i_i64[0];
      v49 = *(_QWORD *)(v46->m128i_i64[0] + 16);
      v50 = *(_QWORD *)(v49 + 40);
      if ( !v46->m128i_i32[2] )
      {
        v47 = sub_87CD20(*(_QWORD *)(v49 + 40), &dword_4F063F8, v84, 0);
        if ( v47 )
          v35 = (const __m128i *)sub_62FD00(v47, 0, 1, (*(_BYTE *)(a1 + 193) & 4) != 0);
        else
          v35 = (const __m128i *)sub_725A70(0);
        if ( dword_4D048B8 )
        {
          v68 = v35;
          v61 = sub_87CF10(v50, v84, &dword_4F063F8);
          if ( v61 )
          {
            v68[1].m128i_i64[0] = v61;
            *(_BYTE *)(v61 + 193) |= 0x40u;
          }
          sub_7340D0(v68, 0, 1);
          v35 = v68;
        }
        v35[3].m128i_i8[1] |= 0x20u;
        *(_QWORD *)(v48 + 24) = v35;
        goto LABEL_68;
      }
      for ( ; *(_BYTE *)(v50 + 140) == 12; v50 = *(_QWORD *)(v50 + 160) )
        ;
      v51 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v50 + 96LL) + 8LL);
      v52 = *(_BYTE *)(v51 + 80);
      if ( v52 != 17 )
        break;
      v51 = *(_QWORD *)(v51 + 88);
      if ( v51 )
        goto LABEL_78;
      v53 = 0;
LABEL_86:
      v57 = sub_6C6630(a1, v53);
      *(_QWORD *)(v48 + 24) = v57;
      v58 = dword_4D048B8;
      *(_BYTE *)(v57 + 49) |= 0x20u;
      if ( v58 && *(_QWORD *)(v57 + 16) )
      {
        ++v46;
        sub_7340D0(v57, 0, 1);
        if ( v83 == v46 )
        {
LABEL_89:
          v1 = v71;
          goto LABEL_90;
        }
      }
      else
      {
LABEL_68:
        if ( v83 == ++v46 )
          goto LABEL_89;
      }
    }
    v53 = *(_QWORD *)(v51 + 88);
    if ( v52 != 20 )
    {
      do
      {
        for ( i = v53; (*(_BYTE *)(i + 194) & 0x40) != 0; i = *(_QWORD *)(i + 232) )
          ;
        if ( v2 == i )
          goto LABEL_86;
LABEL_77:
        v51 = *(_QWORD *)(v51 + 8);
        if ( !v51 )
        {
LABEL_85:
          v53 = 0;
          goto LABEL_86;
        }
LABEL_78:
        v53 = *(_QWORD *)(v51 + 88);
      }
      while ( *(_BYTE *)(v51 + 80) != 20 );
    }
    v55 = *(_QWORD **)(v53 + 168);
    if ( v55 )
    {
      while ( 1 )
      {
        v53 = *(_QWORD *)(v55[3] + 88LL);
        for ( j = v53; (*(_BYTE *)(j + 194) & 0x40) != 0; j = *(_QWORD *)(j + 232) )
          ;
        if ( v2 == j )
          goto LABEL_86;
        v55 = (_QWORD *)*v55;
        if ( !v55 )
        {
          v51 = *(_QWORD *)(v51 + 8);
          if ( v51 )
            goto LABEL_78;
          goto LABEL_85;
        }
      }
    }
    goto LABEL_77;
  }
LABEL_90:
  *v34 = sub_63CAE0(a1, 0, 1, (__int64)v35);
  v59 = v85;
  sub_823A00(v92, 16 * v93);
  sub_823A00(v89, 16 * v90);
  sub_823A00(v86, 16 * v87);
  sub_823A00(v1, v73);
  return v59;
}
