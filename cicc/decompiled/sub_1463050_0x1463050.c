// Function: sub_1463050
// Address: 0x1463050
//
__int64 *__fastcall sub_1463050(
        __int64 **a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 **a6,
        __int64 a7,
        _QWORD *a8,
        _QWORD *a9,
        __int64 *a10,
        __int64 a11)
{
  __int64 v11; // rax
  __int64 **v12; // r13
  __int64 **v13; // r12
  __int64 **v14; // rbx
  __int64 v15; // r14
  __int64 **v16; // r8
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 *v19; // rax
  char *v20; // r8
  __int64 *v21; // r13
  __int64 v22; // r11
  __int64 **v23; // r10
  size_t v24; // r10
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 **v27; // rax
  __int64 **v28; // r13
  __int64 **v29; // r12
  __int64 *result; // rax
  int v31; // eax
  __int64 *v32; // r13
  _QWORD *v33; // r12
  __int64 **v34; // rbx
  __int64 **v35; // rsi
  __int64 **v36; // rdi
  size_t v37; // rdx
  __int64 **v38; // rax
  size_t v39; // r9
  char *v40; // rax
  __int64 **v41; // r14
  char *v42; // rax
  int v43; // [rsp+8h] [rbp-C8h]
  char *v44; // [rsp+8h] [rbp-C8h]
  size_t v45; // [rsp+10h] [rbp-C0h]
  size_t v46; // [rsp+10h] [rbp-C0h]
  int v47; // [rsp+10h] [rbp-C0h]
  int v48; // [rsp+10h] [rbp-C0h]
  size_t v49; // [rsp+10h] [rbp-C0h]
  char *src; // [rsp+18h] [rbp-B8h]
  char *srcb; // [rsp+18h] [rbp-B8h]
  int srcc; // [rsp+18h] [rbp-B8h]
  void *srcd; // [rsp+18h] [rbp-B8h]
  __int64 **srca; // [rsp+18h] [rbp-B8h]
  void *srce; // [rsp+18h] [rbp-B8h]
  int srcf; // [rsp+18h] [rbp-B8h]
  int srcg; // [rsp+18h] [rbp-B8h]
  int srch; // [rsp+18h] [rbp-B8h]
  __int64 *desta; // [rsp+20h] [rbp-B0h]
  __int64 **v61; // [rsp+28h] [rbp-A8h]
  __int64 *v62; // [rsp+28h] [rbp-A8h]
  __int64 **v64; // [rsp+30h] [rbp-A0h]
  __int64 *v65; // [rsp+38h] [rbp-98h]
  void *v66; // [rsp+38h] [rbp-98h]

  v11 = a5;
  v12 = a2;
  v13 = a1;
  v14 = a6;
  if ( a7 <= a5 )
    v11 = a7;
  if ( a4 <= v11 )
  {
LABEL_22:
    if ( v12 != v13 )
      memmove(v14, v13, (char *)v12 - (char *)v13);
    v26 = a11;
    v62 = (__int64 *)((char *)v14 + (char *)v12 - (char *)v13);
    if ( v14 == (__int64 **)v62 || (__int64 **)a3 == v12 )
    {
LABEL_46:
      result = v62;
      if ( v62 != (__int64 *)v14 )
      {
        v35 = v14;
        v36 = v13;
        v37 = (char *)v62 - (char *)v14;
        return (__int64 *)memmove(v36, v35, v37);
      }
    }
    else
    {
      v27 = v12;
      v28 = v13;
      v29 = v27;
      while ( 1 )
      {
        v66 = (void *)v26;
        v31 = sub_1462150(a8, a9, *a10, *v29, (__int64)*v14, v26, 0);
        v26 = (__int64)v66;
        if ( v31 < 0 )
          result = *v29++;
        else
          result = *v14++;
        *v28++ = result;
        if ( v62 == (__int64 *)v14 )
          break;
        if ( (__int64 **)a3 == v29 )
        {
          v13 = v28;
          goto LABEL_46;
        }
      }
    }
  }
  else
  {
    v15 = a5;
    if ( a7 < a5 )
    {
      v65 = (__int64 *)a1;
      v16 = a2;
      v17 = a4;
      while ( 1 )
      {
        src = (char *)v16;
        if ( v17 > v15 )
        {
          v21 = &v65[v17 / 2];
          v38 = sub_1462770(v16, a3, v21, (__int64)a10, (__int64)v16, (__int64)a6, a8, a9, a10, a11);
          v20 = src;
          v22 = v17 / 2;
          v61 = v38;
          v18 = ((char *)v38 - src) >> 3;
        }
        else
        {
          v18 = v15 / 2;
          v61 = &v16[v15 / 2];
          v19 = sub_1462830(v65, (__int64)v16, v61, (__int64)v61, (__int64)v16, (__int64)a6, a8, a9, a10, a11);
          v20 = src;
          v21 = v19;
          v22 = v19 - v65;
        }
        v17 -= v22;
        if ( v17 <= v18 || v18 > a7 )
        {
          if ( v17 > a7 )
          {
            srch = v22;
            v42 = sub_14543A0((char *)v21, v20, (char *)v61);
            LODWORD(v22) = srch;
            v23 = (__int64 **)v42;
          }
          else
          {
            v23 = v61;
            if ( v17 )
            {
              v39 = v20 - (char *)v21;
              if ( v20 != (char *)v21 )
              {
                v44 = v20;
                v48 = v22;
                srce = (void *)(v20 - (char *)v21);
                memmove(a6, v21, v20 - (char *)v21);
                v20 = v44;
                LODWORD(v22) = v48;
                v39 = (size_t)srce;
              }
              if ( v20 != (char *)v61 )
              {
                v49 = v39;
                srcf = v22;
                memmove(v21, v20, (char *)v61 - v20);
                v39 = v49;
                LODWORD(v22) = srcf;
              }
              v23 = (__int64 **)((char *)v61 - v39);
              if ( v39 )
              {
                srcg = v22;
                v40 = (char *)memmove((char *)v61 - v39, a6, v39);
                LODWORD(v22) = srcg;
                v23 = (__int64 **)v40;
              }
            }
          }
        }
        else
        {
          v23 = (__int64 **)v21;
          if ( v18 )
          {
            v24 = (char *)v61 - v20;
            if ( v20 != (char *)v61 )
            {
              v43 = v22;
              v45 = (char *)v61 - v20;
              srcb = v20;
              memmove(a6, v20, (char *)v61 - v20);
              LODWORD(v22) = v43;
              v24 = v45;
              v20 = srcb;
            }
            if ( v20 != (char *)v21 )
            {
              v46 = v24;
              srcc = v22;
              memmove((char *)v61 - (v20 - (char *)v21), v21, v20 - (char *)v21);
              v24 = v46;
              LODWORD(v22) = srcc;
            }
            if ( v24 )
            {
              v47 = v22;
              srcd = (void *)v24;
              memmove(v21, a6, v24);
              LODWORD(v22) = v47;
              v24 = (size_t)srcd;
            }
            v23 = (__int64 **)((char *)v21 + v24);
          }
        }
        v15 -= v18;
        srca = v23;
        sub_1463050(
          (_DWORD)v65,
          (_DWORD)v21,
          (_DWORD)v23,
          v22,
          v18,
          (_DWORD)a6,
          a7,
          (__int64)a8,
          (__int64)a9,
          (__int64)a10,
          a11);
        v25 = a7;
        if ( v15 <= a7 )
          v25 = v15;
        if ( v17 <= v25 )
        {
          v14 = a6;
          v12 = v61;
          v13 = srca;
          goto LABEL_22;
        }
        if ( v15 <= a7 )
          break;
        v65 = (__int64 *)srca;
        v16 = v61;
      }
      v14 = a6;
      v12 = v61;
      v13 = srca;
    }
    if ( (__int64 **)a3 != v12 )
      memmove(v14, v12, a3 - (_QWORD)v12);
    result = (__int64 *)((char *)v14 + a3 - (_QWORD)v12);
    if ( v12 == v13 )
    {
      if ( v14 != (__int64 **)result )
      {
        v37 = a3 - (_QWORD)v12;
        v36 = v12;
        goto LABEL_60;
      }
    }
    else if ( v14 != (__int64 **)result )
    {
      v32 = (__int64 *)(v12 - 1);
      desta = (__int64 *)v13;
      v33 = (_QWORD *)a3;
      v64 = v14;
      v34 = (__int64 **)(result - 1);
      while ( 1 )
      {
        while ( 1 )
        {
          --v33;
          if ( (int)sub_1462150(a8, a9, *a10, *v34, *v32, a11, 0) < 0 )
            break;
          result = *v34;
          *v33 = *v34;
          if ( v64 == v34 )
            return result;
          --v34;
        }
        result = (__int64 *)*v32;
        *v33 = *v32;
        if ( v32 == desta )
          break;
        --v32;
      }
      v41 = v34;
      v14 = v64;
      if ( v64 != v41 + 1 )
      {
        v37 = (char *)(v41 + 1) - (char *)v64;
        v36 = (__int64 **)((char *)v33 - v37);
LABEL_60:
        v35 = v14;
        return (__int64 *)memmove(v36, v35, v37);
      }
    }
  }
  return result;
}
