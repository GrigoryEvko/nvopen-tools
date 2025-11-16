// Function: sub_2B1C790
// Address: 0x2b1c790
//
__int64 __fastcall sub_2B1C790(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        __int64 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v9; // rax
  unsigned int *v10; // r13
  unsigned int *v11; // r12
  unsigned int *v12; // rbx
  __int64 v13; // r15
  unsigned int *v14; // r9
  __int64 v15; // r14
  unsigned int *i; // r11
  __int64 v17; // rbx
  unsigned int *v18; // rax
  int v19; // r11d
  char *v20; // r9
  unsigned int *v21; // r12
  __int64 v22; // rcx
  unsigned int *v23; // r10
  size_t v24; // r10
  __int64 v25; // rax
  __int64 result; // rax
  unsigned int *v27; // r15
  unsigned int *v28; // r15
  unsigned int *v29; // r15
  unsigned int *v30; // r13
  _DWORD *v31; // r12
  unsigned int *v32; // rax
  size_t v33; // r8
  unsigned int *v34; // rax
  size_t v35; // rdx
  unsigned int *v36; // rdi
  char *v37; // rax
  int v38; // [rsp+8h] [rbp-C8h]
  char *v39; // [rsp+8h] [rbp-C8h]
  int v40; // [rsp+10h] [rbp-C0h]
  int v41; // [rsp+10h] [rbp-C0h]
  int v42; // [rsp+10h] [rbp-C0h]
  int v43; // [rsp+10h] [rbp-C0h]
  int v44; // [rsp+10h] [rbp-C0h]
  int v45; // [rsp+10h] [rbp-C0h]
  unsigned int *src; // [rsp+18h] [rbp-B8h]
  void *srca; // [rsp+18h] [rbp-B8h]
  void *srcb; // [rsp+18h] [rbp-B8h]
  int srcc; // [rsp+18h] [rbp-B8h]
  int srcd; // [rsp+18h] [rbp-B8h]
  void *srce; // [rsp+18h] [rbp-B8h]
  int srcf; // [rsp+18h] [rbp-B8h]
  int srcg; // [rsp+18h] [rbp-B8h]
  unsigned int *v54; // [rsp+20h] [rbp-B0h]
  char *v55; // [rsp+20h] [rbp-B0h]
  int v56; // [rsp+20h] [rbp-B0h]
  void *v57; // [rsp+20h] [rbp-B0h]
  unsigned int *v58; // [rsp+20h] [rbp-B0h]
  unsigned int *v59; // [rsp+20h] [rbp-B0h]
  void *v60; // [rsp+20h] [rbp-B0h]
  int v61; // [rsp+20h] [rbp-B0h]
  int v62; // [rsp+20h] [rbp-B0h]
  int v63; // [rsp+20h] [rbp-B0h]
  unsigned int *v65; // [rsp+30h] [rbp-A0h]
  unsigned int *v66; // [rsp+30h] [rbp-A0h]
  __m128i v68; // [rsp+80h] [rbp-50h] BYREF
  __int64 v69; // [rsp+90h] [rbp-40h]

  v9 = a5;
  v10 = a1;
  v11 = a2;
  v12 = a6;
  if ( a7 <= a5 )
    v9 = a7;
  if ( a4 <= v9 )
  {
LABEL_22:
    if ( v11 != v10 )
      memmove(v12, v10, (char *)v11 - (char *)v10);
    result = a9;
    v27 = (unsigned int *)((char *)v12 + (char *)v11 - (char *)v10);
    v69 = a9;
    v68 = _mm_loadu_si128((const __m128i *)&a8);
    if ( v12 != v27 && (unsigned int *)a3 != v11 )
    {
      do
      {
        if ( sub_2B1BC20((__int64 **)&v68, *v11, *v12) )
        {
          result = *v11;
          ++v10;
          ++v11;
          *(v10 - 1) = result;
          if ( v27 == v12 )
            return result;
        }
        else
        {
          result = *v12++;
          *v10++ = result;
          if ( v27 == v12 )
            return result;
        }
      }
      while ( (unsigned int *)a3 != v11 );
    }
    if ( v27 != v12 )
      return (__int64)memmove(v10, v12, (char *)v27 - (char *)v12);
  }
  else
  {
    if ( a7 < a5 )
    {
      v13 = a4;
      v14 = a2;
      v15 = a5;
      for ( i = a1; ; i = v58 )
      {
        if ( v13 > v15 )
        {
          v43 = (int)i;
          v21 = &i[v13 / 2];
          v59 = v14;
          v69 = a9;
          v32 = sub_2B1C1A0(
                  v14,
                  a3,
                  v21,
                  v13 / 2,
                  a5,
                  (__int64)v14,
                  (__int64 *)_mm_loadu_si128((const __m128i *)&a8).m128i_i64[0]);
          v20 = (char *)v59;
          v22 = v13 / 2;
          v65 = v32;
          v19 = v43;
          v17 = v32 - v59;
        }
        else
        {
          src = v14;
          v54 = i;
          v17 = v15 / 2;
          v65 = &v14[v15 / 2];
          v69 = a9;
          v18 = sub_2B1C220(
                  i,
                  (__int64)v14,
                  v65,
                  a4,
                  v15 + ((unsigned __int64)v15 >> 63),
                  (__int64)v14,
                  (__int64 *)_mm_loadu_si128((const __m128i *)&a8).m128i_i64[0]);
          v19 = (int)v54;
          v20 = (char *)src;
          v21 = v18;
          v22 = v18 - v54;
        }
        v13 -= v22;
        if ( v13 <= v17 || v17 > a7 )
        {
          if ( v13 > a7 )
          {
            srcg = v19;
            v63 = v22;
            v37 = sub_2B12540((char *)v21, v20, (char *)v65);
            v19 = srcg;
            LODWORD(v22) = v63;
            v23 = (unsigned int *)v37;
          }
          else
          {
            v23 = v65;
            if ( v13 )
            {
              v33 = v20 - (char *)v21;
              if ( v20 != (char *)v21 )
              {
                v39 = v20;
                v44 = v19;
                srcd = v22;
                v60 = (void *)(v20 - (char *)v21);
                memmove(a6, v21, v20 - (char *)v21);
                v20 = v39;
                v19 = v44;
                LODWORD(v22) = srcd;
                v33 = (size_t)v60;
              }
              if ( v20 != (char *)v65 )
              {
                v45 = v19;
                srce = (void *)v33;
                v61 = v22;
                memmove(v21, v20, (char *)v65 - v20);
                v19 = v45;
                v33 = (size_t)srce;
                LODWORD(v22) = v61;
              }
              v23 = (unsigned int *)((char *)v65 - v33);
              if ( v33 )
              {
                srcf = v19;
                v62 = v22;
                v34 = (unsigned int *)memmove((char *)v65 - v33, a6, v33);
                LODWORD(v22) = v62;
                v19 = srcf;
                v23 = v34;
              }
            }
          }
        }
        else
        {
          v23 = v21;
          if ( v17 )
          {
            v24 = (char *)v65 - v20;
            if ( v20 != (char *)v65 )
            {
              v38 = v19;
              v40 = v22;
              srca = (void *)((char *)v65 - v20);
              v55 = v20;
              memmove(a6, v20, (char *)v65 - v20);
              v19 = v38;
              LODWORD(v22) = v40;
              v24 = (size_t)srca;
              v20 = v55;
            }
            if ( v20 != (char *)v21 )
            {
              v41 = v19;
              srcb = (void *)v24;
              v56 = v22;
              memmove((char *)v65 - (v20 - (char *)v21), v21, v20 - (char *)v21);
              v19 = v41;
              v24 = (size_t)srcb;
              LODWORD(v22) = v56;
            }
            if ( v24 )
            {
              v42 = v19;
              srcc = v22;
              v57 = (void *)v24;
              memmove(v21, a6, v24);
              v19 = v42;
              LODWORD(v22) = srcc;
              v24 = (size_t)v57;
            }
            v23 = (unsigned int *)((char *)v21 + v24);
          }
        }
        v15 -= v17;
        v58 = v23;
        sub_2B1C790(v19, (_DWORD)v21, (_DWORD)v23, v22, v17, (_DWORD)a6, a7, a8, a9);
        v25 = a7;
        if ( v15 <= a7 )
          v25 = v15;
        if ( v13 <= v25 )
        {
          v12 = a6;
          v11 = v65;
          v10 = v58;
          goto LABEL_22;
        }
        if ( v15 <= a7 )
          break;
        v14 = v65;
      }
      v12 = a6;
      v11 = v65;
      v10 = v58;
    }
    if ( (unsigned int *)a3 != v11 )
      memmove(v12, v11, a3 - (_QWORD)v11);
    result = a9;
    v28 = (unsigned int *)((char *)v12 + a3 - (_QWORD)v11);
    v69 = a9;
    v68 = _mm_loadu_si128((const __m128i *)&a8);
    if ( v11 == v10 )
    {
      if ( v12 != v28 )
      {
        v35 = a3 - (_QWORD)v11;
        v36 = v11;
        return (__int64)memmove(v36, v12, v35);
      }
    }
    else if ( v12 != v28 )
    {
      v66 = v10;
      v29 = v28 - 1;
      v30 = v11 - 1;
      v31 = (_DWORD *)a3;
      while ( 1 )
      {
        while ( 1 )
        {
          --v31;
          if ( sub_2B1BC20((__int64 **)&v68, *v29, *v30) )
            break;
          result = *v29;
          *v31 = result;
          if ( v12 == v29 )
            return result;
          --v29;
        }
        result = *v30;
        *v31 = result;
        if ( v30 == v66 )
          break;
        --v30;
      }
      if ( v12 != v29 + 1 )
      {
        v35 = (char *)(v29 + 1) - (char *)v12;
        v36 = (_DWORD *)((char *)v31 - v35);
        return (__int64)memmove(v36, v12, v35);
      }
    }
  }
  return result;
}
