// Function: sub_2DD64B0
// Address: 0x2dd64b0
//
void **__fastcall sub_2DD64B0(char *a1, char *a2, char *a3, __int64 a4, __int64 a5, void **a6, __int64 a7, __int64 a8)
{
  __int64 v8; // rax
  char *v9; // r15
  char *v11; // r13
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // r14
  char *v16; // rax
  char *v17; // r9
  char *v18; // r10
  __int64 v19; // r11
  size_t v20; // rcx
  char *v21; // rax
  __int64 v22; // rax
  signed __int64 v23; // r12
  char *v24; // rax
  char *v25; // r15
  char *v26; // r13
  void **result; // rax
  char v28; // r12
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  char *v31; // r10
  void **v32; // r15
  void ***i; // r14
  void **v34; // r13
  char v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // r12
  void **v38; // rsi
  char *v39; // rdi
  size_t v40; // rdx
  __int64 v41; // rax
  size_t v42; // r8
  unsigned int v43; // eax
  char *v44; // rax
  char *v45; // [rsp+0h] [rbp-80h]
  char *v46; // [rsp+0h] [rbp-80h]
  int v47; // [rsp+8h] [rbp-78h]
  size_t v48; // [rsp+8h] [rbp-78h]
  int v49; // [rsp+8h] [rbp-78h]
  int v50; // [rsp+8h] [rbp-78h]
  int v51; // [rsp+8h] [rbp-78h]
  size_t v52; // [rsp+10h] [rbp-70h]
  int v53; // [rsp+10h] [rbp-70h]
  int v54; // [rsp+10h] [rbp-70h]
  size_t v55; // [rsp+10h] [rbp-70h]
  size_t v56; // [rsp+10h] [rbp-70h]
  int v57; // [rsp+10h] [rbp-70h]
  int v58; // [rsp+10h] [rbp-70h]
  void **dest; // [rsp+18h] [rbp-68h]
  char *desta; // [rsp+18h] [rbp-68h]
  char *v61; // [rsp+20h] [rbp-60h]
  void **v62; // [rsp+20h] [rbp-60h]
  void **v63; // [rsp+20h] [rbp-60h]
  char *v64; // [rsp+28h] [rbp-58h]
  void **v65; // [rsp+28h] [rbp-58h]
  void *v66; // [rsp+28h] [rbp-58h]
  void *v67; // [rsp+28h] [rbp-58h]
  char *src; // [rsp+30h] [rbp-50h]
  char *srca; // [rsp+30h] [rbp-50h]
  char *srcb; // [rsp+30h] [rbp-50h]
  char *srcc; // [rsp+30h] [rbp-50h]
  void *srcd; // [rsp+30h] [rbp-50h]
  __int64 srce; // [rsp+30h] [rbp-50h]
  __int64 srcf; // [rsp+30h] [rbp-50h]
  char *srcg; // [rsp+30h] [rbp-50h]
  int srch; // [rsp+30h] [rbp-50h]

  v8 = a5;
  v9 = a2;
  v11 = a1;
  if ( a7 <= a5 )
    v8 = a7;
  if ( a4 <= v8 )
  {
LABEL_22:
    v23 = v9 - v11;
    if ( v11 != v9 )
      memmove(a6, v11, v9 - v11);
    v62 = (void **)((char *)a6 + v23);
    if ( a6 != (void **)((char *)a6 + v23) && a3 != v9 )
    {
      v24 = v9;
      v25 = v11;
      v26 = v24;
      do
      {
        v65 = (void **)*a6;
        srce = *(_QWORD *)(*(_QWORD *)v26 + 24LL);
        v28 = sub_AE5020(a8, srce);
        v29 = sub_9208B0(a8, srce);
        v66 = v65[3];
        v30 = ((1LL << v28) + ((unsigned __int64)(v29 + 7) >> 3) - 1) >> v28 << v28;
        LOBYTE(srce) = sub_AE5020(a8, (__int64)v66);
        if ( ((1LL << srce) + ((unsigned __int64)(sub_9208B0(a8, (__int64)v66) + 7) >> 3) - 1) >> srce << srce > v30 )
        {
          result = *(void ***)v26;
          v25 += 8;
          v26 += 8;
          *((_QWORD *)v25 - 1) = result;
          if ( v62 == a6 )
            return result;
        }
        else
        {
          result = (void **)*a6;
          v25 += 8;
          ++a6;
          *((_QWORD *)v25 - 1) = result;
          if ( v62 == a6 )
            return result;
        }
      }
      while ( a3 != v26 );
      v11 = v25;
    }
    result = v62;
    if ( v62 != a6 )
    {
      v38 = a6;
      v39 = v11;
      v40 = (char *)v62 - (char *)a6;
      return (void **)memmove(v39, v38, v40);
    }
  }
  else
  {
    if ( a7 < a5 )
    {
      v64 = a1;
      v12 = (__int64)a2;
      v13 = a5;
      v14 = a4;
      dest = a6;
      while ( 1 )
      {
        src = (char *)v12;
        if ( v14 > v13 )
        {
          v41 = sub_2DD5A60(v12, (__int64)a3, (__int64 *)&v64[8 * (v14 / 2)], a8);
          v17 = src;
          v18 = &v64[8 * (v14 / 2)];
          v61 = (char *)v41;
          v19 = v14 / 2;
          v15 = (v41 - (__int64)src) >> 3;
        }
        else
        {
          v15 = v13 / 2;
          v61 = (char *)(v12 + 8 * (v13 / 2));
          v16 = (char *)sub_2DD5BA0(v64, v12, (__int64)v61, a8);
          v17 = src;
          v18 = v16;
          v19 = (v16 - v64) >> 3;
        }
        v14 -= v19;
        if ( v14 <= v15 || v15 > a7 )
        {
          if ( v14 > a7 )
          {
            v51 = v19;
            v58 = (int)v18;
            v44 = sub_2DD4030(v18, v17, v61);
            LODWORD(v19) = v51;
            LODWORD(v18) = v58;
            srca = v44;
          }
          else
          {
            srca = v61;
            if ( v14 )
            {
              v42 = v17 - v18;
              if ( v17 != v18 )
              {
                v46 = v17;
                v49 = v19;
                v55 = v17 - v18;
                srcg = v18;
                memmove(dest, v18, v17 - v18);
                v17 = v46;
                LODWORD(v19) = v49;
                v42 = v55;
                v18 = srcg;
              }
              if ( v17 != v61 )
              {
                v56 = v42;
                srch = v19;
                v43 = (unsigned int)memmove(v18, v17, v61 - v17);
                v42 = v56;
                LODWORD(v19) = srch;
                LODWORD(v18) = v43;
              }
              srca = &v61[-v42];
              if ( v42 )
              {
                v50 = (int)v18;
                v57 = v19;
                memmove(&v61[-v42], dest, v42);
                LODWORD(v19) = v57;
                LODWORD(v18) = v50;
              }
            }
          }
        }
        else
        {
          srca = v18;
          if ( v15 )
          {
            v20 = v61 - v17;
            if ( v17 != v61 )
            {
              v45 = v18;
              v47 = v19;
              v52 = v61 - v17;
              srcb = v17;
              memmove(dest, v17, v61 - v17);
              v18 = v45;
              LODWORD(v19) = v47;
              v20 = v52;
              v17 = srcb;
            }
            if ( v17 != v18 )
            {
              v48 = v20;
              v53 = v19;
              srcc = v18;
              memmove(&v61[-(v17 - v18)], v18, v17 - v18);
              v20 = v48;
              LODWORD(v19) = v53;
              v18 = srcc;
            }
            if ( v20 )
            {
              v54 = v19;
              srcd = (void *)v20;
              v21 = (char *)memmove(v18, dest, v20);
              LODWORD(v19) = v54;
              v20 = (size_t)srcd;
              v18 = v21;
            }
            srca = &v18[v20];
          }
        }
        v13 -= v15;
        sub_2DD64B0((_DWORD)v64, (_DWORD)v18, (_DWORD)srca, v19, v15, (_DWORD)dest, a7, a8);
        v22 = a7;
        if ( v13 <= a7 )
          v22 = v13;
        if ( v14 <= v22 )
        {
          a6 = dest;
          v9 = v61;
          v11 = srca;
          goto LABEL_22;
        }
        if ( v13 <= a7 )
          break;
        v12 = (__int64)v61;
        v64 = srca;
      }
      a6 = dest;
      v9 = v61;
      v11 = srca;
    }
    if ( a3 != v9 )
      memmove(a6, v9, a3 - v9);
    result = (void **)((char *)a6 + a3 - v9);
    if ( v11 == v9 )
    {
      if ( a6 != result )
      {
        v40 = a3 - v9;
        v39 = v9;
        goto LABEL_59;
      }
    }
    else if ( a6 != result )
    {
      desta = v11;
      v31 = v9 - 8;
      v32 = result - 1;
      v63 = a6;
      for ( i = (void ***)v31; ; --i )
      {
        while ( 1 )
        {
          v34 = *i;
          srcf = *((_QWORD *)*v32 + 3);
          v35 = sub_AE5020(a8, srcf);
          v36 = sub_9208B0(a8, srcf);
          v67 = v34[3];
          v37 = (((unsigned __int64)(v36 + 7) >> 3) + (1LL << v35) - 1) >> v35 << v35;
          LOBYTE(srcf) = sub_AE5020(a8, (__int64)v67);
          a3 -= 8;
          if ( (((unsigned __int64)(sub_9208B0(a8, (__int64)v67) + 7) >> 3) + (1LL << srcf) - 1) >> srcf << srcf > v37 )
            break;
          result = (void **)*v32;
          *(_QWORD *)a3 = *v32;
          if ( v63 == v32 )
            return result;
          --v32;
        }
        result = *i;
        *(_QWORD *)a3 = *i;
        if ( desta == (char *)i )
          break;
      }
      a6 = v63;
      if ( v63 != v32 + 1 )
      {
        v40 = (char *)(v32 + 1) - (char *)v63;
        v39 = &a3[-v40];
LABEL_59:
        v38 = a6;
        return (void **)memmove(v39, v38, v40);
      }
    }
  }
  return result;
}
