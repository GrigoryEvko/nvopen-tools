// Function: sub_2BB7660
// Address: 0x2bb7660
//
void *__fastcall sub_2BB7660(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, void **a6, void *a7, __int128 a8)
{
  __int64 v8; // r15
  char *v9; // r13
  char *v10; // r12
  __int128 v12; // xmm0
  void *result; // rax
  __int64 v14; // r14
  char *v15; // r14
  char *v16; // r9
  __int64 v17; // r12
  __int64 v18; // r13
  char *v19; // rax
  char *v20; // r9
  char *v21; // r10
  __int64 v22; // r11
  size_t v23; // rcx
  char *v24; // rax
  void **v25; // r15
  size_t v26; // rdx
  void **v27; // r15
  void **v28; // r15
  char *v29; // r13
  _QWORD *v30; // r12
  void **v31; // rsi
  char *v32; // rdi
  char *v33; // rax
  size_t v34; // r8
  unsigned int v35; // eax
  _QWORD *v36; // rdi
  char *v37; // rax
  char *v38; // [rsp+8h] [rbp-88h]
  char *v39; // [rsp+8h] [rbp-88h]
  int v40; // [rsp+10h] [rbp-80h]
  size_t v41; // [rsp+10h] [rbp-80h]
  int v42; // [rsp+10h] [rbp-80h]
  int v43; // [rsp+10h] [rbp-80h]
  int v44; // [rsp+10h] [rbp-80h]
  size_t v45; // [rsp+18h] [rbp-78h]
  int v46; // [rsp+18h] [rbp-78h]
  int v47; // [rsp+18h] [rbp-78h]
  size_t v48; // [rsp+18h] [rbp-78h]
  size_t v49; // [rsp+18h] [rbp-78h]
  int v50; // [rsp+18h] [rbp-78h]
  int v51; // [rsp+18h] [rbp-78h]
  char *v52; // [rsp+28h] [rbp-68h]
  char *src; // [rsp+38h] [rbp-58h]
  char *srca; // [rsp+38h] [rbp-58h]
  char *srcb; // [rsp+38h] [rbp-58h]
  char *srcc; // [rsp+38h] [rbp-58h]
  void *srcd; // [rsp+38h] [rbp-58h]
  char *srce; // [rsp+38h] [rbp-58h]
  int srcf; // [rsp+38h] [rbp-58h]
  char *v60; // [rsp+40h] [rbp-50h]

  v8 = a4;
  v9 = a1;
  v10 = a2;
  v12 = (__int128)_mm_loadu_si128((const __m128i *)&a8);
  result = (void *)a5;
  v14 = *((_QWORD *)&v12 + 1);
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( a4 <= (__int64)result )
  {
LABEL_22:
    if ( v10 != v9 )
      result = memmove(a6, v9, v10 - v9);
    v25 = (void **)((char *)a6 + v10 - v9);
    if ( a6 != v25 )
    {
      while ( (char *)a3 != v10 )
      {
        if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, void *))v12)(v14, *(_QWORD *)v10, *a6) )
        {
          result = *(void **)v10;
          v9 += 8;
          v10 += 8;
          *((_QWORD *)v9 - 1) = result;
          if ( v25 == a6 )
            return result;
        }
        else
        {
          result = *a6++;
          v9 += 8;
          *((_QWORD *)v9 - 1) = result;
          if ( v25 == a6 )
            return result;
        }
      }
    }
    if ( v25 != a6 )
    {
      v31 = a6;
      v32 = v9;
      v26 = (char *)v25 - (char *)a6;
      return memmove(v32, v31, v26);
    }
  }
  else
  {
    if ( (__int64)a7 < a5 )
    {
      v15 = a1;
      v16 = a2;
      v17 = a5;
      while ( 1 )
      {
        src = v16;
        if ( v8 > v17 )
        {
          v33 = (char *)sub_2BB72F0(
                          v16,
                          a3,
                          &v15[8 * (v8 / 2)],
                          (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v12,
                          *((__int64 *)&v12 + 1));
          v20 = src;
          v21 = &v15[8 * (v8 / 2)];
          v52 = v33;
          v22 = v8 / 2;
          v18 = (v33 - src) >> 3;
        }
        else
        {
          v18 = v17 / 2;
          v52 = &v16[8 * (v17 / 2)];
          v19 = (char *)sub_2BB7370(
                          v15,
                          (__int64)v16,
                          v52,
                          (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v12,
                          *((__int64 *)&v12 + 1));
          v20 = src;
          v21 = v19;
          v22 = (v19 - v15) >> 3;
        }
        v8 -= v22;
        if ( v8 <= v18 || v18 > (__int64)a7 )
        {
          if ( v8 > (__int64)a7 )
          {
            v44 = v22;
            v51 = (int)v21;
            v37 = sub_2B12380(v21, v20, v52);
            LODWORD(v22) = v44;
            LODWORD(v21) = v51;
            srca = v37;
          }
          else
          {
            srca = v52;
            if ( v8 )
            {
              v34 = v20 - v21;
              if ( v20 != v21 )
              {
                v42 = v22;
                v39 = v20;
                v48 = v20 - v21;
                srce = v21;
                memmove(a6, v21, v20 - v21);
                v20 = v39;
                LODWORD(v22) = v42;
                v34 = v48;
                v21 = srce;
              }
              if ( v20 != v52 )
              {
                v49 = v34;
                srcf = v22;
                v35 = (unsigned int)memmove(v21, v20, v52 - v20);
                v34 = v49;
                LODWORD(v22) = srcf;
                LODWORD(v21) = v35;
              }
              srca = &v52[-v34];
              if ( v34 )
              {
                v43 = (int)v21;
                v50 = v22;
                memmove(&v52[-v34], a6, v34);
                LODWORD(v22) = v50;
                LODWORD(v21) = v43;
              }
            }
          }
        }
        else
        {
          srca = v21;
          if ( v18 )
          {
            v23 = v52 - v20;
            if ( v20 != v52 )
            {
              v40 = v22;
              v38 = v21;
              v45 = v52 - v20;
              srcb = v20;
              memmove(a6, v20, v52 - v20);
              v21 = v38;
              LODWORD(v22) = v40;
              v23 = v45;
              v20 = srcb;
            }
            if ( v20 != v21 )
            {
              v41 = v23;
              v46 = v22;
              srcc = v21;
              memmove(&v52[-(v20 - v21)], v21, v20 - v21);
              v23 = v41;
              LODWORD(v22) = v46;
              v21 = srcc;
            }
            if ( v23 )
            {
              v47 = v22;
              srcd = (void *)v23;
              v24 = (char *)memmove(v21, a6, v23);
              LODWORD(v22) = v47;
              v23 = (size_t)srcd;
              v21 = v24;
            }
            srca = &v21[v23];
          }
        }
        v17 -= v18;
        sub_2BB7660((_DWORD)v15, (_DWORD)v21, (_DWORD)srca, v22, v18, (_DWORD)a6, (__int64)a7, v12);
        result = a7;
        if ( v17 <= (__int64)a7 )
          result = (void *)v17;
        if ( v8 <= (__int64)result )
        {
          v14 = *((_QWORD *)&v12 + 1);
          v10 = v52;
          v9 = srca;
          goto LABEL_22;
        }
        if ( v17 <= (__int64)a7 )
          break;
        v16 = v52;
        v15 = srca;
      }
      v14 = *((_QWORD *)&v12 + 1);
      v10 = v52;
      v9 = srca;
    }
    result = (void *)a3;
    v26 = a3 - (_QWORD)v10;
    if ( (char *)a3 != v10 )
    {
      result = memmove(a6, v10, v26);
      v26 = a3 - (_QWORD)v10;
    }
    v27 = (void **)((char *)a6 + v26);
    if ( v10 == v9 )
    {
      if ( a6 != v27 )
      {
        v36 = (_QWORD *)a3;
        goto LABEL_57;
      }
    }
    else if ( a6 != v27 )
    {
      v60 = v9;
      v28 = v27 - 1;
      v29 = v10 - 8;
      v30 = (_QWORD *)a3;
      while ( 1 )
      {
        while ( 1 )
        {
          --v30;
          if ( ((unsigned __int8 (__fastcall *)(__int64, void *, _QWORD))v12)(v14, *v28, *(_QWORD *)v29) )
            break;
          result = *v28;
          *v30 = *v28;
          if ( a6 == v28 )
            return result;
          --v28;
        }
        result = *(void **)v29;
        *v30 = *(_QWORD *)v29;
        if ( v29 == v60 )
          break;
        v29 -= 8;
      }
      if ( a6 != v28 + 1 )
      {
        v26 = (char *)(v28 + 1) - (char *)a6;
        v36 = v30;
LABEL_57:
        v32 = (char *)v36 - v26;
        v31 = a6;
        return memmove(v32, v31, v26);
      }
    }
  }
  return result;
}
