// Function: sub_23FC720
// Address: 0x23fc720
//
__int64 **__fastcall sub_23FC720(
        __int64 ***a1,
        __int64 ***a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 ***a6,
        __int64 a7,
        __int64 a8)
{
  __int64 ***v8; // r13
  __int64 ***v9; // r12
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 ***v14; // r9
  __int64 v15; // rbx
  __int64 ***v16; // rax
  char *v17; // r9
  __int64 ***v18; // r12
  __int64 v19; // r11
  __int64 ***v20; // r10
  size_t v21; // r10
  __int64 v22; // rax
  signed __int64 v23; // r15
  __int64 ***v24; // r15
  __int64 **result; // rax
  __int64 **v26; // r13
  unsigned int v27; // r14d
  __int64 ***v28; // r15
  __int64 ***v29; // r13
  _QWORD *v30; // r12
  __int64 ***v31; // rbx
  __int64 **v32; // r15
  unsigned int v33; // r14d
  __int64 ***v34; // rsi
  __int64 ***v35; // rdi
  size_t v36; // rdx
  __int64 ***v37; // rax
  size_t v38; // r8
  char *v39; // rax
  __int64 ***v40; // r15
  char *v41; // rax
  int v42; // [rsp+0h] [rbp-70h]
  char *v43; // [rsp+0h] [rbp-70h]
  size_t v44; // [rsp+8h] [rbp-68h]
  size_t v45; // [rsp+8h] [rbp-68h]
  int v46; // [rsp+8h] [rbp-68h]
  int v47; // [rsp+8h] [rbp-68h]
  size_t v48; // [rsp+8h] [rbp-68h]
  char *src; // [rsp+10h] [rbp-60h]
  char *srcb; // [rsp+10h] [rbp-60h]
  int srcc; // [rsp+10h] [rbp-60h]
  void *srcd; // [rsp+10h] [rbp-60h]
  __int64 ***srca; // [rsp+10h] [rbp-60h]
  void *srce; // [rsp+10h] [rbp-60h]
  int srcf; // [rsp+10h] [rbp-60h]
  int srcg; // [rsp+10h] [rbp-60h]
  int srch; // [rsp+10h] [rbp-60h]
  __int64 ***dest; // [rsp+18h] [rbp-58h]
  __int64 ***v59; // [rsp+20h] [rbp-50h]
  __int64 ***v60; // [rsp+28h] [rbp-48h]
  __int64 **v61; // [rsp+30h] [rbp-40h]
  __int64 ***v62; // [rsp+30h] [rbp-40h]
  __int64 ***v64; // [rsp+38h] [rbp-38h]

  v8 = a1;
  v9 = a2;
  v11 = a5;
  if ( a7 <= a5 )
    v11 = a7;
  if ( a4 <= v11 )
  {
LABEL_22:
    v23 = (char *)v9 - (char *)v8;
    if ( v8 != v9 )
      memmove(a6, v8, (char *)v9 - (char *)v8);
    v61 = (__int64 **)((char *)a6 + v23);
    if ( a6 != (__int64 ***)((char *)a6 + v23) && (__int64 ***)a3 != v9 )
    {
      v24 = v8;
      do
      {
        v26 = *a6;
        v27 = sub_22DADF0(***v9);
        if ( v27 < (unsigned int)sub_22DADF0(**v26) )
        {
          result = *v9;
          ++v24;
          ++v9;
          *(v24 - 1) = result;
          if ( v61 == (__int64 **)a6 )
            return result;
        }
        else
        {
          result = *a6;
          ++v24;
          ++a6;
          *(v24 - 1) = result;
          if ( v61 == (__int64 **)a6 )
            return result;
        }
      }
      while ( (__int64 ***)a3 != v9 );
      v8 = v24;
    }
    result = v61;
    if ( v61 != (__int64 **)a6 )
    {
      v34 = a6;
      v35 = v8;
      v36 = (char *)v61 - (char *)a6;
      return (__int64 **)memmove(v35, v34, v36);
    }
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v60 = a1;
      v13 = a4;
      v14 = a2;
      dest = a6;
      while ( 1 )
      {
        src = (char *)v14;
        if ( v12 < v13 )
        {
          v18 = &v60[v13 / 2];
          v37 = sub_23FB300(v14, a3, v18);
          v17 = src;
          v19 = v13 / 2;
          v59 = v37;
          v15 = ((char *)v37 - src) >> 3;
        }
        else
        {
          v15 = v12 / 2;
          v59 = &v14[v12 / 2];
          v16 = sub_23FB260(v60, (__int64)v14, v59);
          v17 = src;
          v18 = v16;
          v19 = v16 - v60;
        }
        v13 -= v19;
        if ( v13 <= v15 || a7 < v15 )
        {
          if ( a7 < v13 )
          {
            srch = v19;
            v41 = sub_23FB780((char *)v18, v17, (char *)v59);
            LODWORD(v19) = srch;
            v20 = (__int64 ***)v41;
          }
          else
          {
            v20 = v59;
            if ( v13 )
            {
              v38 = v17 - (char *)v18;
              if ( v17 != (char *)v18 )
              {
                v43 = v17;
                v47 = v19;
                srce = (void *)(v17 - (char *)v18);
                memmove(dest, v18, v17 - (char *)v18);
                v17 = v43;
                LODWORD(v19) = v47;
                v38 = (size_t)srce;
              }
              if ( v17 != (char *)v59 )
              {
                v48 = v38;
                srcf = v19;
                memmove(v18, v17, (char *)v59 - v17);
                v38 = v48;
                LODWORD(v19) = srcf;
              }
              v20 = (__int64 ***)((char *)v59 - v38);
              if ( v38 )
              {
                srcg = v19;
                v39 = (char *)memmove((char *)v59 - v38, dest, v38);
                LODWORD(v19) = srcg;
                v20 = (__int64 ***)v39;
              }
            }
          }
        }
        else
        {
          v20 = v18;
          if ( v15 )
          {
            v21 = (char *)v59 - v17;
            if ( v17 != (char *)v59 )
            {
              v42 = v19;
              v44 = (char *)v59 - v17;
              srcb = v17;
              memmove(dest, v17, (char *)v59 - v17);
              LODWORD(v19) = v42;
              v21 = v44;
              v17 = srcb;
            }
            if ( v17 != (char *)v18 )
            {
              v45 = v21;
              srcc = v19;
              memmove((char *)v59 - (v17 - (char *)v18), v18, v17 - (char *)v18);
              v21 = v45;
              LODWORD(v19) = srcc;
            }
            if ( v21 )
            {
              v46 = v19;
              srcd = (void *)v21;
              memmove(v18, dest, v21);
              LODWORD(v19) = v46;
              v21 = (size_t)srcd;
            }
            v20 = (__int64 ***)((char *)v18 + v21);
          }
        }
        v12 -= v15;
        srca = v20;
        sub_23FC720((_DWORD)v60, (_DWORD)v18, (_DWORD)v20, v19, v15, (_DWORD)dest, a7, a8);
        v22 = v12;
        if ( a7 <= v12 )
          v22 = a7;
        if ( v22 >= v13 )
        {
          a6 = dest;
          v9 = v59;
          v8 = srca;
          goto LABEL_22;
        }
        if ( a7 >= v12 )
          break;
        v60 = srca;
        v14 = v59;
      }
      a6 = dest;
      v9 = v59;
      v8 = srca;
    }
    result = (__int64 **)a3;
    if ( (__int64 ***)a3 != v9 )
      result = (__int64 **)memmove(a6, v9, a3 - (_QWORD)v9);
    v28 = (__int64 ***)((char *)a6 + a3 - (_QWORD)v9);
    if ( v9 == v8 )
    {
      if ( a6 != v28 )
      {
        v36 = a3 - (_QWORD)v9;
        v35 = v9;
        goto LABEL_59;
      }
    }
    else if ( a6 != v28 )
    {
      v62 = v8;
      v29 = v9 - 1;
      v30 = (_QWORD *)a3;
      v64 = a6;
      v31 = v28 - 1;
      while ( 1 )
      {
        while ( 1 )
        {
          v32 = *v29;
          --v30;
          v33 = sub_22DADF0(***v31);
          if ( v33 < (unsigned int)sub_22DADF0(**v32) )
            break;
          result = *v31;
          *v30 = *v31;
          if ( v64 == v31 )
            return result;
          --v31;
        }
        result = *v29;
        *v30 = *v29;
        if ( v29 == v62 )
          break;
        --v29;
      }
      v40 = v31;
      a6 = v64;
      if ( v64 != v40 + 1 )
      {
        v36 = (char *)(v40 + 1) - (char *)v64;
        v35 = (__int64 ***)((char *)v30 - v36);
LABEL_59:
        v34 = a6;
        return (__int64 **)memmove(v35, v34, v36);
      }
    }
  }
  return result;
}
