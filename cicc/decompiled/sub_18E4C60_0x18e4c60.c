// Function: sub_18E4C60
// Address: 0x18e4c60
//
__int64 __fastcall sub_18E4C60(__int64 *a1, __int64 a2, unsigned int a3, __m128i a4, __m128i a5)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // r15
  char v13; // al
  unsigned int v14; // esi
  unsigned __int64 v15; // r13
  __int64 *v16; // r14
  __int64 v17; // r15
  _QWORD *v18; // r12
  __int64 *v19; // r13
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rax
  __int64 *v24; // r8
  __int64 *v25; // rdi
  __int64 *v26; // rax
  unsigned int v27; // esi
  __int64 *v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rbx
  __int64 v31; // rax
  unsigned int v32; // ebx
  __int64 v33; // rax
  _QWORD *v34; // rbx
  __int64 v35; // rax
  unsigned int v36; // r14d
  __int64 *v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // [rsp+0h] [rbp-250h]
  unsigned __int8 v47; // [rsp+17h] [rbp-239h]
  __int64 v48; // [rsp+18h] [rbp-238h]
  int v51; // [rsp+3Ch] [rbp-214h] BYREF
  __int64 v52; // [rsp+40h] [rbp-210h] BYREF
  __int64 v53; // [rsp+48h] [rbp-208h] BYREF
  __int64 v54; // [rsp+50h] [rbp-200h] BYREF
  __int64 v55; // [rsp+58h] [rbp-1F8h] BYREF
  _BYTE *v56; // [rsp+60h] [rbp-1F0h] BYREF
  __int64 v57; // [rsp+68h] [rbp-1E8h]
  _BYTE v58[128]; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v59; // [rsp+F0h] [rbp-160h] BYREF
  __int64 *v60; // [rsp+F8h] [rbp-158h]
  __int64 *v61; // [rsp+100h] [rbp-150h]
  __int64 v62; // [rsp+108h] [rbp-148h]
  int v63; // [rsp+110h] [rbp-140h]
  _BYTE v64[312]; // [rsp+118h] [rbp-138h] BYREF

  v47 = sub_18E4B00(a1, a2, a3, &v52, &v53, &v54, a4, a5);
  if ( !v47 )
    return v47;
  if ( (unsigned int)*(unsigned __int8 *)(v52 + 16) - 9 <= 7 )
    return 0;
  v5 = sub_146F1B0(*a1, v52);
  v62 = 32;
  v48 = v5;
  v60 = (__int64 *)v64;
  v61 = (__int64 *)v64;
  v56 = v58;
  v57 = 0x1000000000LL;
  v63 = 0;
  v59 = 0;
  v6 = *(_QWORD *)(v52 + 8);
  if ( !v6 )
    goto LABEL_33;
  do
  {
    while ( 1 )
    {
      v7 = sub_1648700(v6);
      v8 = v7;
      if ( (_QWORD *)a2 != v7 && *((_BYTE *)v7 + 16) > 0x17u && (unsigned __int8)sub_14AFF20(a2, (__int64)v7, a1[1]) )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_12;
    }
    v11 = (unsigned int)v57;
    if ( (unsigned int)v57 >= HIDWORD(v57) )
    {
      sub_16CD150((__int64)&v56, v58, 0, 8, v9, v10);
      v11 = (unsigned int)v57;
    }
    *(_QWORD *)&v56[8 * v11] = v8;
    LODWORD(v57) = v57 + 1;
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v6 );
LABEL_12:
  while ( (_DWORD)v57 )
  {
    v12 = *(_QWORD *)&v56[8 * (unsigned int)v57 - 8];
    LODWORD(v57) = v57 - 1;
    v13 = *(_BYTE *)(v12 + 16);
    switch ( v13 )
    {
      case '6':
        v14 = sub_18E4500(v48, v53, v54, *(_QWORD *)(v12 - 24), (_QWORD *)*a1, a4, a5);
        if ( v14 > 1 << (*(unsigned __int16 *)(v12 + 18) >> 1) >> 1 )
          sub_15F8F50(v12, v14);
        break;
      case '7':
        v27 = sub_18E4500(v48, v53, v54, *(_QWORD *)(v12 - 24), (_QWORD *)*a1, a4, a5);
        if ( v27 > 1 << (*(unsigned __int16 *)(v12 + 18) >> 1) >> 1 )
          sub_15F9450(v12, v27);
        break;
      case 'N':
        v29 = *(_QWORD *)(v12 - 24);
        if ( !*(_BYTE *)(v29 + 16)
          && (*(_BYTE *)(v29 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v29 + 36) - 133) <= 4
          && ((1LL << (*(_BYTE *)(v29 + 36) + 123)) & 0x15) != 0 )
        {
          v30 = (_QWORD *)*a1;
          v31 = sub_1649C60(*(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
          v32 = sub_18E4500(v48, v53, v54, v31, v30, a4, a5);
          if ( v32 > (unsigned int)sub_15603A0((_QWORD *)(v12 + 56), 0) )
          {
            v55 = *(_QWORD *)(v12 + 56);
            v42 = (__int64 *)sub_16498A0(v12);
            *(_QWORD *)(v12 + 56) = sub_1563C10(&v55, v42, 1, 1);
            if ( v32 )
            {
              v43 = (__int64 *)sub_16498A0(v12);
              v44 = sub_155D330(v43, v32);
              v51 = 0;
              v46 = v44;
              v55 = *(_QWORD *)(v12 + 56);
              v45 = (__int64 *)sub_16498A0(v12);
              v55 = sub_1563E10(&v55, v45, &v51, 1, v46);
              *(_QWORD *)(v12 + 56) = v55;
            }
          }
          v33 = *(_QWORD *)(v12 - 24);
          if ( *(_BYTE *)(v33 + 16) )
            BUG();
          if ( (*(_DWORD *)(v33 + 36) & 0xFFFFFFFD) == 0x85 )
          {
            v34 = (_QWORD *)*a1;
            v35 = sub_1649C60(*(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF))));
            v36 = sub_18E4500(v48, v53, v54, v35, v34, a4, a5);
            if ( v36 > (unsigned int)sub_15603A0((_QWORD *)(v12 + 56), 1) )
            {
              v55 = *(_QWORD *)(v12 + 56);
              v37 = (__int64 *)sub_16498A0(v12);
              *(_QWORD *)(v12 + 56) = sub_1563C10(&v55, v37, 2, 1);
              if ( v36 )
              {
                v38 = (__int64 *)sub_16498A0(v12);
                v39 = sub_155D330(v38, v36);
                v51 = 1;
                v40 = v39;
                v55 = *(_QWORD *)(v12 + 56);
                v41 = (__int64 *)sub_16498A0(v12);
                v55 = sub_1563E10(&v55, v41, &v51, 1, v40);
                *(_QWORD *)(v12 + 56) = v55;
              }
            }
          }
        }
        break;
    }
    v15 = (unsigned __int64)v61;
    v16 = v60;
    if ( v61 != v60 )
      goto LABEL_17;
    v24 = &v61[HIDWORD(v62)];
    if ( v61 != v24 )
    {
      v25 = 0;
      v26 = v61;
      while ( v12 != *v26 )
      {
        if ( *v26 == -2 )
          v25 = v26;
        if ( v24 == ++v26 )
        {
          if ( !v25 )
            goto LABEL_70;
          *v25 = v12;
          v15 = (unsigned __int64)v61;
          --v63;
          v16 = v60;
          ++v59;
          goto LABEL_18;
        }
      }
      goto LABEL_18;
    }
LABEL_70:
    if ( HIDWORD(v62) < (unsigned int)v62 )
    {
      ++HIDWORD(v62);
      *v24 = v12;
      v16 = v60;
      ++v59;
      v15 = (unsigned __int64)v61;
    }
    else
    {
LABEL_17:
      sub_16CCBA0((__int64)&v59, v12);
      v15 = (unsigned __int64)v61;
      v16 = v60;
    }
LABEL_18:
    v17 = *(_QWORD *)(v12 + 8);
    if ( v17 )
    {
      while ( 1 )
      {
        v18 = sub_1648700(v17);
        if ( v16 == (__int64 *)v15 )
        {
          v19 = &v16[HIDWORD(v62)];
          if ( v16 == v19 )
          {
            v28 = v16;
          }
          else
          {
            do
            {
              if ( v18 == (_QWORD *)*v16 )
                break;
              ++v16;
            }
            while ( v19 != v16 );
            v28 = v19;
          }
        }
        else
        {
          v19 = (__int64 *)(v15 + 8LL * (unsigned int)v62);
          v16 = sub_16CC9F0((__int64)&v59, (__int64)v18);
          if ( v18 == (_QWORD *)*v16 )
          {
            if ( v61 == v60 )
              v28 = &v61[HIDWORD(v62)];
            else
              v28 = &v61[(unsigned int)v62];
          }
          else
          {
            if ( v61 != v60 )
            {
              v16 = &v61[(unsigned int)v62];
              goto LABEL_26;
            }
            v16 = &v61[HIDWORD(v62)];
            v28 = v16;
          }
        }
        while ( v28 != v16 && (unsigned __int64)*v16 >= 0xFFFFFFFFFFFFFFFELL )
          ++v16;
LABEL_26:
        if ( v19 == v16 && (unsigned __int8)sub_14AFF20(a2, (__int64)v18, a1[1]) )
        {
          v22 = (unsigned int)v57;
          if ( (unsigned int)v57 >= HIDWORD(v57) )
          {
            sub_16CD150((__int64)&v56, v58, 0, 8, v20, v21);
            v22 = (unsigned int)v57;
          }
          *(_QWORD *)&v56[8 * v22] = v18;
          LODWORD(v57) = v57 + 1;
        }
        v17 = *(_QWORD *)(v17 + 8);
        if ( !v17 )
          goto LABEL_12;
        v15 = (unsigned __int64)v61;
        v16 = v60;
      }
    }
  }
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
LABEL_33:
  if ( v61 != v60 )
    _libc_free((unsigned __int64)v61);
  return v47;
}
