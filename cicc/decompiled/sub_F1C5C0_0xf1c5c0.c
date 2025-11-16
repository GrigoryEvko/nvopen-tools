// Function: sub_F1C5C0
// Address: 0xf1c5c0
//
__int64 __fastcall sub_F1C5C0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 **a7)
{
  __int64 *v8; // rbx
  __int64 v9; // r8
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 *v14; // rbx
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 *v19; // r12
  __int64 *v20; // rbx
  __int64 *v21; // rsi
  unsigned __int64 v22; // rax
  __int64 *v23; // r14
  __int64 v24; // r13
  __int64 *i; // rbx
  __int64 v26; // rdi
  __int64 *v27; // r15
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int8 v34; // dl
  __int64 v35; // r15
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r8
  int v42; // eax
  __int64 *v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 result; // rax
  unsigned __int8 **v47; // rdx
  __int64 v48; // rax
  _BYTE *v49; // rbx
  _BYTE *v50; // r13
  _QWORD *v51; // rdi
  __int64 v52; // [rsp+0h] [rbp-200h]
  __int64 *src; // [rsp+20h] [rbp-1E0h]
  unsigned __int8 *v56; // [rsp+38h] [rbp-1C8h]
  _BYTE v57[32]; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 *v58; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v59; // [rsp+68h] [rbp-198h]
  _BYTE v60[16]; // [rsp+70h] [rbp-190h] BYREF
  __int64 *v61; // [rsp+80h] [rbp-180h] BYREF
  __int64 v62; // [rsp+88h] [rbp-178h]
  _BYTE v63[16]; // [rsp+90h] [rbp-170h] BYREF
  _BYTE *v64; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v65; // [rsp+A8h] [rbp-158h]
  _BYTE v66[16]; // [rsp+B0h] [rbp-150h] BYREF
  __m128i v67; // [rsp+C0h] [rbp-140h] BYREF
  char v68; // [rsp+D8h] [rbp-128h]
  __int64 v69; // [rsp+E0h] [rbp-120h]
  _QWORD v70[2]; // [rsp+F0h] [rbp-110h] BYREF
  _BYTE v71[168]; // [rsp+100h] [rbp-100h] BYREF
  int v72; // [rsp+1A8h] [rbp-58h] BYREF
  __int64 v73; // [rsp+1B0h] [rbp-50h]
  int *v74; // [rsp+1B8h] [rbp-48h]
  int *v75; // [rsp+1C0h] [rbp-40h]
  __int64 v76; // [rsp+1C8h] [rbp-38h]

  v8 = *a7;
  v56 = a2;
  v9 = (__int64)&(*a7)[*((unsigned int *)a7 + 2)];
  v58 = (__int64 *)v60;
  v59 = 0x200000000LL;
  if ( v8 == (__int64 *)v9 )
  {
    v20 = (__int64 *)v63;
    v62 = 0x200000000LL;
    v61 = (__int64 *)v63;
    v19 = (__int64 *)v63;
  }
  else
  {
    v11 = 0;
    v12 = v9;
    do
    {
      v13 = *v8;
      if ( a6 != *(_QWORD *)(*v8 + 40) )
      {
        if ( v11 + 1 > (unsigned __int64)HIDWORD(v59) )
        {
          a2 = v60;
          sub_C8D5F0((__int64)&v58, v60, v11 + 1, 8u, v9, a6);
          v11 = (unsigned int)v59;
        }
        v58[v11] = v13;
        v11 = (unsigned int)(v59 + 1);
        LODWORD(v59) = v59 + 1;
      }
      ++v8;
    }
    while ( (__int64 *)v12 != v8 );
    v14 = v58;
    v15 = &v58[v11];
    v61 = (__int64 *)v63;
    v62 = 0x200000000LL;
    if ( v15 == v58 )
    {
      v19 = (__int64 *)v63;
      v20 = (__int64 *)v63;
    }
    else
    {
      v16 = 0;
      do
      {
        while ( 1 )
        {
          v17 = *v14;
          if ( a5 == *(_QWORD *)(*v14 + 40) )
            break;
          if ( v15 == ++v14 )
            goto LABEL_15;
        }
        if ( v16 + 1 > (unsigned __int64)HIDWORD(v62) )
        {
          a2 = v63;
          sub_C8D5F0((__int64)&v61, v63, v16 + 1, 8u, v9, a6);
          v16 = (unsigned int)v62;
        }
        ++v14;
        v61[v16] = v17;
        v16 = (unsigned int)(v62 + 1);
        LODWORD(v62) = v62 + 1;
      }
      while ( v15 != v14 );
LABEL_15:
      v18 = 8 * v16;
      v19 = &v61[v16];
      v20 = v19;
      if ( v61 != v19 )
      {
        v21 = &v61[v16];
        src = v61;
        _BitScanReverse64(&v22, v18 >> 3);
        sub_F06D00(v61, v21, 2LL * (int)(63 - (v22 ^ 0x3F)));
        if ( (unsigned __int64)v18 <= 0x80 )
        {
          a2 = (unsigned __int8 *)v19;
          sub_F06860(src, v19);
        }
        else
        {
          v23 = src + 16;
          a2 = (unsigned __int8 *)(src + 16);
          sub_F06860(src, src + 16);
          if ( v19 != src + 16 )
          {
            do
            {
              v24 = *v23;
              for ( i = v23; ; i[1] = *i )
              {
                v26 = *(i - 1);
                a2 = (unsigned __int8 *)v24;
                v27 = i--;
                if ( !sub_B445A0(v26, v24) )
                  break;
              }
              ++v23;
              *v27 = v24;
            }
            while ( v19 != v23 );
          }
        }
        v19 = v61;
        v20 = &v61[(unsigned int)v62];
      }
    }
  }
  v72 = 0;
  v64 = v66;
  v65 = 0x200000000LL;
  v70[0] = v71;
  v70[1] = 0x400000000LL;
  v73 = 0;
  v74 = &v72;
  v75 = &v72;
  v76 = 0;
  if ( v19 == v20 )
  {
    v45 = 0;
    goto LABEL_51;
  }
  do
  {
    while ( 1 )
    {
      v28 = *v19;
      v29 = *(_QWORD *)(*v19 - 32);
      if ( !v29 || *(_BYTE *)v29 || *(_QWORD *)(v29 + 24) != *(_QWORD *)(v28 + 80) )
        BUG();
      if ( *(_DWORD *)(v29 + 36) == 69 )
        goto LABEL_48;
      v30 = sub_B10CD0(v28 + 48);
      v34 = *(_BYTE *)(v30 - 16);
      if ( (v34 & 2) == 0 )
      {
        v31 = (*(_WORD *)(v30 - 16) >> 6) & 0xF;
        if ( ((*(_WORD *)(v30 - 16) >> 6) & 0xF) != 2 )
          goto LABEL_32;
        v48 = v30 - 16 - 8LL * ((v34 >> 2) & 0xF);
LABEL_68:
        v35 = *(_QWORD *)(v48 + 8);
        goto LABEL_33;
      }
      if ( *(_DWORD *)(v30 - 24) == 2 )
      {
        v48 = *(_QWORD *)(v30 - 32);
        goto LABEL_68;
      }
LABEL_32:
      v35 = 0;
LABEL_33:
      v36 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
      v37 = *(_QWORD *)(*(_QWORD *)(v28 + 32 * (2 - v36)) + 24LL);
      v67.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v28 + 32 * (1 - v36)) + 24LL);
      if ( v37 )
        sub_AF47B0((__int64)&v67.m128i_i64[1], *(unsigned __int64 **)(v37 + 16), *(unsigned __int64 **)(v37 + 24));
      else
        v68 = 0;
      v69 = v35;
      a2 = (unsigned __int8 *)v70;
      sub_F1C2C0((__int64)v57, (__int64)v70, &v67, v31, v32, v33);
      if ( v57[16] )
      {
        v38 = *(_QWORD *)(v28 - 32);
        if ( !v38 || *(_BYTE *)v38 || *(_QWORD *)(v38 + 24) != *(_QWORD *)(v28 + 80) )
          BUG();
        if ( *(_DWORD *)(v38 + 36) != 68 )
        {
          v39 = sub_B47F80((_BYTE *)v28);
          v40 = (unsigned int)v65;
          v41 = v39;
          v42 = v65;
          if ( (unsigned int)v65 >= (unsigned __int64)HIDWORD(v65) )
          {
            if ( HIDWORD(v65) < (unsigned __int64)(unsigned int)v65 + 1 )
            {
              a2 = v66;
              v52 = v41;
              sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 8u, v41, (unsigned int)v65 + 1LL);
              v40 = (unsigned int)v65;
              v41 = v52;
            }
            *(_QWORD *)&v64[8 * v40] = v41;
            LODWORD(v65) = v65 + 1;
          }
          else
          {
            v43 = (__int64 *)&v64[8 * (unsigned int)v65];
            if ( v43 )
            {
              *v43 = v41;
              v42 = v65;
            }
            LODWORD(v65) = v42 + 1;
          }
          v44 = *(_QWORD *)(v28 - 32);
          if ( !v44 || *(_BYTE *)v44 || *(_QWORD *)(v44 + 24) != *(_QWORD *)(v28 + 80) )
            BUG();
          if ( *(_DWORD *)(v44 + 36) == 69 && (unsigned int)*v56 - 67 <= 0xC )
            break;
        }
      }
LABEL_48:
      if ( v20 == ++v19 )
        goto LABEL_49;
    }
    if ( (v56[7] & 0x40) != 0 )
      v47 = (unsigned __int8 **)*((_QWORD *)v56 - 1);
    else
      v47 = (unsigned __int8 **)&v56[-32 * (*((_DWORD *)v56 + 1) & 0x7FFFFFF)];
    a2 = v56;
    ++v19;
    sub_B59720(*(_QWORD *)&v64[8 * (unsigned int)v65 - 8], (__int64)v56, *v47);
  }
  while ( v20 != v19 );
LABEL_49:
  if ( (_DWORD)v65 )
  {
    a2 = (unsigned __int8 *)v58;
    sub_F54050(v56, v58, (unsigned int)v59, 0, 0);
    v49 = v64;
    v50 = &v64[8 * (unsigned int)v65];
    if ( v64 != v50 )
    {
      do
      {
        v51 = (_QWORD *)*((_QWORD *)v50 - 1);
        a2 = (unsigned __int8 *)a3;
        v50 -= 8;
        sub_B44220(v51, a3, a4);
      }
      while ( v49 != v50 );
    }
  }
  v45 = v73;
LABEL_51:
  result = sub_F078B0(v45);
  if ( (_BYTE *)v70[0] != v71 )
    result = _libc_free(v70[0], a2);
  if ( v64 != v66 )
    result = _libc_free(v64, a2);
  if ( v61 != (__int64 *)v63 )
    result = _libc_free(v61, a2);
  if ( v58 != (__int64 *)v60 )
    return _libc_free(v58, a2);
  return result;
}
