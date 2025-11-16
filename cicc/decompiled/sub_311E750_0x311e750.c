// Function: sub_311E750
// Address: 0x311e750
//
unsigned __int64 *__fastcall sub_311E750(
        unsigned __int64 **a1,
        unsigned __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 **a6,
        __int64 a7,
        unsigned __int64 *a8)
{
  unsigned __int64 **v8; // r13
  unsigned __int64 **v9; // r12
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rax
  char *v17; // r9
  unsigned __int64 **v18; // r13
  __int64 v19; // r11
  unsigned __int64 **v20; // r10
  size_t v21; // r10
  __int64 v22; // rax
  unsigned __int64 *result; // rax
  unsigned __int64 **v24; // r14
  unsigned __int64 **v25; // r14
  unsigned __int64 **v26; // r14
  unsigned __int64 **v27; // r13
  _QWORD *v28; // r12
  __int64 v29; // rax
  size_t v30; // r8
  char *v31; // rax
  size_t v32; // rdx
  unsigned __int64 **v33; // rdi
  char *v34; // rax
  int v35; // [rsp+0h] [rbp-80h]
  char *v36; // [rsp+0h] [rbp-80h]
  size_t v37; // [rsp+8h] [rbp-78h]
  size_t v38; // [rsp+8h] [rbp-78h]
  int v39; // [rsp+8h] [rbp-78h]
  int v40; // [rsp+8h] [rbp-78h]
  size_t v41; // [rsp+8h] [rbp-78h]
  char *src; // [rsp+10h] [rbp-70h]
  char *srcb; // [rsp+10h] [rbp-70h]
  int srcc; // [rsp+10h] [rbp-70h]
  void *srcd; // [rsp+10h] [rbp-70h]
  unsigned __int64 **srca; // [rsp+10h] [rbp-70h]
  void *srce; // [rsp+10h] [rbp-70h]
  int srcf; // [rsp+10h] [rbp-70h]
  int srcg; // [rsp+10h] [rbp-70h]
  int srch; // [rsp+10h] [rbp-70h]
  unsigned __int64 **dest; // [rsp+18h] [rbp-68h]
  unsigned __int64 **v52; // [rsp+20h] [rbp-60h]
  unsigned __int64 **v53; // [rsp+28h] [rbp-58h]
  unsigned __int64 **v54; // [rsp+30h] [rbp-50h]
  __int64 v56[7]; // [rsp+48h] [rbp-38h] BYREF

  v8 = a1;
  v9 = a2;
  v11 = a5;
  if ( a7 <= a5 )
    v11 = a7;
  if ( a4 <= v11 )
  {
LABEL_22:
    if ( v9 != v8 )
      memmove(a6, v8, (char *)v9 - (char *)v8);
    result = a8;
    v24 = (unsigned __int64 **)((char *)a6 + (char *)v9 - (char *)v8);
    v56[0] = (__int64)a8;
    if ( a6 != v24 && (unsigned __int64 **)a3 != v9 )
    {
      do
      {
        if ( (unsigned __int8)sub_311D9B0(v56, v9, a6) )
        {
          result = *v9;
          ++v8;
          ++v9;
          *(v8 - 1) = result;
          if ( v24 == a6 )
            return result;
        }
        else
        {
          result = *a6++;
          *v8++ = result;
          if ( v24 == a6 )
            return result;
        }
      }
      while ( (unsigned __int64 **)a3 != v9 );
    }
    if ( v24 != a6 )
      return (unsigned __int64 *)memmove(v8, a6, (char *)v24 - (char *)a6);
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v53 = a1;
      v13 = (__int64)a2;
      v14 = a4;
      dest = a6;
      while ( 1 )
      {
        src = (char *)v13;
        if ( v14 > v12 )
        {
          v18 = &v53[v14 / 2];
          v29 = sub_311E0B0(v13, a3, v18, (__int64)a8);
          v17 = src;
          v19 = v14 / 2;
          v52 = (unsigned __int64 **)v29;
          v15 = (v29 - (__int64)src) >> 3;
        }
        else
        {
          v15 = v12 / 2;
          v52 = (unsigned __int64 **)(v13 + 8 * (v12 / 2));
          v16 = sub_311E130((__int64)v53, v13, v52, (__int64)a8);
          v17 = src;
          v18 = (unsigned __int64 **)v16;
          v19 = (v16 - (__int64)v53) >> 3;
        }
        v14 -= v19;
        if ( v14 <= v15 || v15 > a7 )
        {
          if ( v14 > a7 )
          {
            srch = v19;
            v34 = sub_311D7F0((char *)v18, v17, (char *)v52);
            LODWORD(v19) = srch;
            v20 = (unsigned __int64 **)v34;
          }
          else
          {
            v20 = v52;
            if ( v14 )
            {
              v30 = v17 - (char *)v18;
              if ( v17 != (char *)v18 )
              {
                v36 = v17;
                v40 = v19;
                srce = (void *)(v17 - (char *)v18);
                memmove(dest, v18, v17 - (char *)v18);
                v17 = v36;
                LODWORD(v19) = v40;
                v30 = (size_t)srce;
              }
              if ( v17 != (char *)v52 )
              {
                v41 = v30;
                srcf = v19;
                memmove(v18, v17, (char *)v52 - v17);
                v30 = v41;
                LODWORD(v19) = srcf;
              }
              v20 = (unsigned __int64 **)((char *)v52 - v30);
              if ( v30 )
              {
                srcg = v19;
                v31 = (char *)memmove((char *)v52 - v30, dest, v30);
                LODWORD(v19) = srcg;
                v20 = (unsigned __int64 **)v31;
              }
            }
          }
        }
        else
        {
          v20 = v18;
          if ( v15 )
          {
            v21 = (char *)v52 - v17;
            if ( v17 != (char *)v52 )
            {
              v35 = v19;
              v37 = (char *)v52 - v17;
              srcb = v17;
              memmove(dest, v17, (char *)v52 - v17);
              LODWORD(v19) = v35;
              v21 = v37;
              v17 = srcb;
            }
            if ( v17 != (char *)v18 )
            {
              v38 = v21;
              srcc = v19;
              memmove((char *)v52 - (v17 - (char *)v18), v18, v17 - (char *)v18);
              v21 = v38;
              LODWORD(v19) = srcc;
            }
            if ( v21 )
            {
              v39 = v19;
              srcd = (void *)v21;
              memmove(v18, dest, v21);
              LODWORD(v19) = v39;
              v21 = (size_t)srcd;
            }
            v20 = (unsigned __int64 **)((char *)v18 + v21);
          }
        }
        v12 -= v15;
        srca = v20;
        sub_311E750((_DWORD)v53, (_DWORD)v18, (_DWORD)v20, v19, v15, (_DWORD)dest, a7, (__int64)a8);
        v22 = a7;
        if ( v12 <= a7 )
          v22 = v12;
        if ( v14 <= v22 )
        {
          a6 = dest;
          v9 = v52;
          v8 = srca;
          goto LABEL_22;
        }
        if ( v12 <= a7 )
          break;
        v53 = srca;
        v13 = (__int64)v52;
      }
      a6 = dest;
      v9 = v52;
      v8 = srca;
    }
    if ( (unsigned __int64 **)a3 != v9 )
      memmove(a6, v9, a3 - (_QWORD)v9);
    result = a8;
    v25 = (unsigned __int64 **)((char *)a6 + a3 - (_QWORD)v9);
    v56[0] = (__int64)a8;
    if ( v9 == v8 )
    {
      if ( a6 != v25 )
      {
        v32 = a3 - (_QWORD)v9;
        v33 = v9;
        return (unsigned __int64 *)memmove(v33, a6, v32);
      }
    }
    else if ( a6 != v25 )
    {
      v54 = v8;
      v26 = v25 - 1;
      v27 = v9 - 1;
      v28 = (_QWORD *)a3;
      while ( 1 )
      {
        while ( 1 )
        {
          --v28;
          if ( (unsigned __int8)sub_311D9B0(v56, v26, v27) )
            break;
          result = *v26;
          *v28 = *v26;
          if ( a6 == v26 )
            return result;
          --v26;
        }
        result = *v27;
        *v28 = *v27;
        if ( v27 == v54 )
          break;
        --v27;
      }
      if ( a6 != v26 + 1 )
      {
        v32 = (char *)(v26 + 1) - (char *)a6;
        v33 = (unsigned __int64 **)((char *)v28 - v32);
        return (unsigned __int64 *)memmove(v33, a6, v32);
      }
    }
  }
  return result;
}
