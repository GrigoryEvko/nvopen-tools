// Function: sub_FDFC80
// Address: 0xfdfc80
//
__int64 __fastcall sub_FDFC80(__int64 a1, _QWORD *a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r14
  char *v8; // rax
  char *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int16 *v12; // rax
  unsigned __int16 *v13; // rsi
  __int64 v14; // r14
  char v15; // r12
  unsigned int v16; // r15d
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned __int16 *v23; // r13
  __int64 v24; // r10
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rcx
  _QWORD *v28; // rdx
  _QWORD *v29; // r12
  _QWORD *v30; // r15
  __int64 v31; // r13
  _QWORD *i; // r14
  __int64 v33; // r15
  _QWORD *v34; // r13
  __int64 v35; // rax
  __int64 v36; // rsi
  __int16 v37; // cx
  __int64 v38; // rdi
  _QWORD *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rcx
  unsigned int v43; // ecx
  __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned int v46; // esi
  __int64 *v47; // rax
  __int64 v48; // r10
  __int64 v49; // rax
  char *v50; // r12
  __int64 v51; // r13
  __int64 v52; // rbx
  _QWORD *v53; // rdi
  __int64 result; // rax
  char *j; // rbx
  char v56; // dl
  unsigned int v57; // eax
  signed __int64 v58; // rax
  unsigned __int16 v59; // dx
  unsigned __int16 v60; // r12
  __int64 v61; // rdx
  unsigned __int16 v62; // cx
  __int16 v63; // dx
  unsigned __int64 v64; // rax
  int v65; // eax
  int v66; // r9d
  __int64 v67; // [rsp+8h] [rbp-148h]
  __int64 v68; // [rsp+10h] [rbp-140h]
  char *v69; // [rsp+18h] [rbp-138h]
  __int64 v71; // [rsp+28h] [rbp-128h]
  char *v72; // [rsp+30h] [rbp-120h]
  __int64 v74; // [rsp+58h] [rbp-F8h]
  __int64 v75; // [rsp+60h] [rbp-F0h]
  __int64 v77; // [rsp+70h] [rbp-E0h]
  __int64 v78; // [rsp+70h] [rbp-E0h]
  __int64 v79; // [rsp+78h] [rbp-D8h]
  char *v80; // [rsp+78h] [rbp-D8h]
  _QWORD *v81; // [rsp+80h] [rbp-D0h]
  __int64 v82; // [rsp+80h] [rbp-D0h]
  __int64 v83; // [rsp+80h] [rbp-D0h]
  int v84; // [rsp+88h] [rbp-C8h]
  __int64 v85; // [rsp+88h] [rbp-C8h]
  unsigned __int16 v86; // [rsp+9Ch] [rbp-B4h] BYREF
  unsigned __int16 v87; // [rsp+9Eh] [rbp-B2h] BYREF
  unsigned __int16 *v88; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int64 v89; // [rsp+A8h] [rbp-A8h] BYREF
  __m128i v90; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned __int16 v91; // [rsp+C0h] [rbp-90h]
  __int64 v92; // [rsp+D0h] [rbp-80h] BYREF
  unsigned int v93; // [rsp+D8h] [rbp-78h]
  int v94; // [rsp+E8h] [rbp-68h]
  __m128i v95; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v96; // [rsp+100h] [rbp-50h]
  int v97; // [rsp+108h] [rbp-48h]
  char v98; // [rsp+10Ch] [rbp-44h]
  char v99; // [rsp+110h] [rbp-40h] BYREF

  v5 = a2[1] - *a2;
  v75 = v5 >> 3;
  if ( (unsigned __int64)v5 > 0x2AAAAAAAAAAAAAA8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v6 = 16 * v75;
  v67 = 3 * v75;
  v7 = 24 * v75;
  if ( !v75 )
  {
    v68 = 0;
    v27 = 0;
    v26 = 0;
    v71 = 0;
    v69 = 0;
    v72 = 0;
    goto LABEL_30;
  }
  v8 = (char *)sub_22077B0(24 * v75);
  v72 = v8;
  v9 = &v8[v7];
  v69 = &v8[v7];
  do
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = 0;
      *((_QWORD *)v8 + 1) = 0;
      *((_QWORD *)v8 + 2) = 0;
    }
    v8 += 24;
  }
  while ( v8 != v9 );
  v10 = sub_22077B0(v6);
  v11 = v10 + v6;
  v71 = v10;
  v68 = v10 + v6;
  do
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *(_WORD *)(v10 + 8) = 0;
    }
    v10 += 16;
  }
  while ( v10 != v11 );
  v79 = 0;
  v77 = v71;
  v74 = (__int64)v72;
  do
  {
    v12 = *(unsigned __int16 **)(*a2 + 8 * v79);
    v13 = (unsigned __int16 *)&v88;
    v98 = 1;
    v95.m128i_i64[0] = 0;
    v88 = v12;
    v96 = 2;
    v95.m128i_i64[1] = (__int64)&v99;
    v97 = 0;
    sub_FDC9F0((__int64)&v92, &v88);
    v14 = v92;
    v15 = v98;
    v84 = v94;
    if ( v93 != v94 )
    {
      v16 = v93;
      while ( 1 )
      {
        v17 = sub_B46EC0(v14, v16);
        v13 = *(unsigned __int16 **)(a3 + 8);
        v19 = v17;
        v20 = *(unsigned int *)(a3 + 24);
        if ( !(_DWORD)v20 )
          goto LABEL_22;
        v21 = ((_DWORD)v20 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v22 = 16 * v21;
        v23 = &v13[8 * v21];
        v24 = *(_QWORD *)v23;
        if ( v19 != *(_QWORD *)v23 )
        {
          v22 = 1;
          while ( v24 != -4096 )
          {
            v18 = (unsigned int)(v22 + 1);
            v21 = ((_DWORD)v20 - 1) & (unsigned int)(v22 + v21);
            v23 = &v13[8 * (unsigned int)v21];
            v24 = *(_QWORD *)v23;
            if ( v19 == *(_QWORD *)v23 )
              goto LABEL_16;
            v22 = (unsigned int)v18;
          }
          goto LABEL_22;
        }
LABEL_16:
        if ( v23 == &v13[8 * v20] )
          goto LABEL_22;
        if ( v15 )
        {
          v25 = (_QWORD *)v95.m128i_i64[1];
          v13 = (unsigned __int16 *)HIDWORD(v96);
          v21 = v95.m128i_i64[1] + 8LL * HIDWORD(v96);
          if ( v95.m128i_i64[1] != v21 )
          {
            while ( v19 != *v25 )
            {
              if ( (_QWORD *)v21 == ++v25 )
                goto LABEL_67;
            }
            goto LABEL_22;
          }
LABEL_67:
          if ( HIDWORD(v96) < (unsigned int)v96 )
          {
            ++HIDWORD(v96);
            *(_QWORD *)v21 = v19;
            ++v95.m128i_i64[0];
            goto LABEL_60;
          }
        }
        v13 = (unsigned __int16 *)v19;
        v82 = v19;
        sub_C8CC70((__int64)&v95, v19, v21, v22, v19, v18);
        v15 = v98;
        v19 = v82;
        if ( v56 )
        {
LABEL_60:
          v13 = v88;
          v57 = sub_FF0430(*(_QWORD *)(a1 + 112), v88, v19);
          if ( v57 )
          {
            v58 = sub_F04200(v57, 0x80000000);
            v60 = v59;
            v61 = *((_QWORD *)v23 + 1);
            v90.m128i_i64[1] = v58;
            v83 = v58;
            v90.m128i_i64[0] = v61;
            v91 = v60;
            sub_FDFC40(v74, &v90);
            v87 = v60;
            v62 = *(_WORD *)(v77 + 8);
            v89 = *(_QWORD *)v77;
            v13 = &v86;
            v86 = v62;
            v90.m128i_i64[0] = v83;
            v63 = sub_FDCA70(&v89, &v86, (unsigned __int64 *)&v90, &v87);
            v64 = v89 + v90.m128i_i64[0];
            if ( __CFADD__(v89, v90.m128i_i64[0]) )
            {
              ++v63;
              v64 = (v64 >> 1) | 0x8000000000000000LL;
            }
            *(_QWORD *)v77 = v64;
            *(_WORD *)(v77 + 8) = v63;
            if ( v63 > 0x3FFF )
            {
              *(_QWORD *)v77 = -1;
              *(_WORD *)(v77 + 8) = 0x3FFF;
            }
          }
          v15 = v98;
          if ( v84 == ++v16 )
            break;
        }
        else
        {
LABEL_22:
          if ( v84 == ++v16 )
            break;
        }
      }
    }
    if ( !v15 )
      _libc_free(v95.m128i_i64[1], v13);
    ++v79;
    v77 += 16;
    v74 += 24;
  }
  while ( v79 != v75 );
  v26 = (_QWORD *)sub_22077B0(v67 * 8);
  v27 = v26;
  v28 = &v26[v67];
  do
  {
    if ( v26 )
    {
      *v26 = 0;
      v26[1] = 0;
      v26[2] = 0;
    }
    v26 += 3;
  }
  while ( v28 != v26 );
LABEL_30:
  v29 = (_QWORD *)*a4;
  v30 = (_QWORD *)a4[1];
  *a4 = v27;
  v31 = a4[2];
  a4[1] = v26;
  a4[2] = v26;
  for ( i = v29; v30 != i; i += 3 )
  {
    if ( *i )
      j_j___libc_free_0(*i, i[2] - *i);
  }
  if ( v29 )
    j_j___libc_free_0(v29, v31 - (_QWORD)v29);
  v85 = 0;
  v33 = v71;
  v80 = v72;
  if ( v75 )
  {
    v78 = a3;
    do
    {
      v34 = *(_QWORD **)v80;
      v81 = (_QWORD *)*((_QWORD *)v80 + 1);
      if ( v81 != *(_QWORD **)v80 )
      {
        do
        {
          v35 = *v34;
          v36 = v34[1];
          v34 += 3;
          v37 = *((_WORD *)v34 - 4);
          v38 = 3 * v35;
          v39 = (_QWORD *)*a4;
          v95.m128i_i64[0] = v36;
          v95.m128i_i16[4] = v37;
          v40 = (__int64)&v39[v38];
          v41 = sub_FDE760((__int64)&v95, v33);
          v42 = *(_QWORD *)v41;
          LOWORD(v41) = *(_WORD *)(v41 + 8);
          v95.m128i_i64[0] = v85;
          v95.m128i_i64[1] = v42;
          LOWORD(v96) = v41;
          sub_FDFC40(v40, &v95);
        }
        while ( v81 != v34 );
      }
      ++v85;
      v33 += 16;
      v80 += 24;
    }
    while ( v85 != v75 );
    a3 = v78;
  }
  v43 = *(_DWORD *)(a3 + 24);
  v44 = *(_QWORD *)(a3 + 8);
  v45 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + 80LL);
  if ( v45 )
    v45 -= 24;
  if ( v43 )
  {
    v46 = (v43 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v47 = (__int64 *)(v44 + 16LL * v46);
    v48 = *v47;
    if ( v45 == *v47 )
      goto LABEL_46;
    v65 = 1;
    while ( v48 != -4096 )
    {
      v66 = v65 + 1;
      v46 = (v43 - 1) & (v65 + v46);
      v47 = (__int64 *)(v44 + 16LL * v46);
      v48 = *v47;
      if ( v45 == *v47 )
        goto LABEL_46;
      v65 = v66;
    }
  }
  v47 = (__int64 *)(v44 + 16LL * v43);
LABEL_46:
  v49 = v47[1];
  if ( v75 )
  {
    v50 = v72;
    v51 = 0;
    v52 = 3 * v49;
    do
    {
      while ( *((_QWORD *)v50 + 1) != *(_QWORD *)v50 )
      {
        ++v51;
        v50 += 24;
        if ( v51 == v75 )
          goto LABEL_51;
      }
      v95.m128i_i64[0] = v51++;
      v95.m128i_i64[1] = 1;
      v50 += 24;
      v53 = (_QWORD *)*a4;
      LOWORD(v96) = 0;
      sub_FDFC40((__int64)&v53[v52], &v95);
    }
    while ( v51 != v75 );
  }
LABEL_51:
  if ( v71 )
    j_j___libc_free_0(v71, v68 - v71);
  result = (__int64)v72;
  for ( j = v72; v69 != j; j += 24 )
  {
    if ( *(_QWORD *)j )
      result = j_j___libc_free_0(*(_QWORD *)j, *((_QWORD *)j + 2) - *(_QWORD *)j);
  }
  if ( v72 )
    return j_j___libc_free_0(v72, v69 - v72);
  return result;
}
