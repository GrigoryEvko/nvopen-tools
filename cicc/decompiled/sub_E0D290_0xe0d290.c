// Function: sub_E0D290
// Address: 0xe0d290
//
char *__fastcall sub_E0D290(void **a1, unsigned __int64 *a2, unsigned __int64 a3)
{
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  size_t v8; // rdx
  unsigned __int64 v9; // rax
  char *v10; // r8
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  char *result; // rax
  size_t v14; // r15
  char *v15; // rdi
  unsigned __int64 v16; // rdx
  void *v17; // r8
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  int v21; // ecx
  size_t v22; // r9
  char *v23; // rsi
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  size_t v26; // rdx
  unsigned __int64 v27; // rax
  char *v28; // r8
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  size_t v33; // rdx
  unsigned __int64 v34; // rax
  char *v35; // r8
  unsigned __int64 v36; // rax
  __int64 v37; // rax

  v6 = a2[1];
  v7 = *a2;
  if ( a3 == 11 )
  {
    if ( v7 <= 0xB )
      goto LABEL_15;
    if ( *(_QWORD *)v6 == 0x667265746E495F5FLL && *(_DWORD *)(v6 + 8) == 1516594017 )
    {
      v33 = (size_t)a1[1];
      v34 = (unsigned __int64)a1[2];
      v35 = (char *)*a1;
      if ( v34 < v33 + 14 )
      {
        v36 = 2 * v34;
        if ( v33 + 1006 > v36 )
          a1[2] = (void *)(v33 + 1006);
        else
          a1[2] = (void *)v36;
        v37 = realloc(v35);
        *a1 = (void *)v37;
        v35 = (char *)v37;
        if ( !v37 )
          goto LABEL_76;
        v33 = (size_t)a1[1];
      }
      memmove(v35 + 14, v35, v33);
      result = (char *)*a1;
      qmemcpy(*a1, "Interface for ", 14);
      a1[1] = (char *)a1[1] + 13;
      a2[1] += 11LL;
      *a2 -= 11LL;
      return result;
    }
LABEL_38:
    v15 = (char *)a1[1];
    v16 = (unsigned __int64)a1[2];
    v17 = *a1;
LABEL_39:
    if ( a3 <= v7 )
      v7 = a3;
    v14 = v7;
    goto LABEL_20;
  }
  if ( a3 > 0xB )
  {
    if ( a3 != 12 || v7 <= 0xC )
      goto LABEL_15;
    if ( *(_QWORD *)v6 == 0x656C75646F4D5F5FLL && *(_DWORD *)(v6 + 8) == 1868983881 && *(_BYTE *)(v6 + 12) == 90 )
    {
      v26 = (size_t)a1[1];
      v27 = (unsigned __int64)a1[2];
      v28 = (char *)*a1;
      if ( v26 + 15 > v27 )
      {
        v29 = 2 * v27;
        if ( v26 + 1007 > v29 )
          a1[2] = (void *)(v26 + 1007);
        else
          a1[2] = (void *)v29;
        v30 = realloc(v28);
        *a1 = (void *)v30;
        v28 = (char *)v30;
        if ( !v30 )
          goto LABEL_76;
        v26 = (size_t)a1[1];
      }
      memmove(v28 + 15, v28, v26);
      result = (char *)*a1;
      qmemcpy(*a1, "ModuleInfo for ", 15);
      a1[1] = (char *)a1[1] + 14;
      a2[1] += 12LL;
      *a2 -= 12LL;
      return result;
    }
    goto LABEL_38;
  }
  if ( a3 != 6 )
  {
    if ( a3 == 7 && v7 > 7 )
    {
      if ( *(_QWORD *)v6 == 0x5A7373616C435F5FLL )
      {
        v8 = (size_t)a1[1];
        v9 = (unsigned __int64)a1[2];
        v10 = (char *)*a1;
        if ( v9 >= v8 + 14 )
        {
LABEL_12:
          memmove(v10 + 14, v10, v8);
          result = (char *)*a1;
          qmemcpy(*a1, "ClassInfo for ", 14);
          a1[1] = (char *)a1[1] + 13;
          a2[1] += 7LL;
          *a2 -= 7LL;
          return result;
        }
        v11 = 2 * v9;
        if ( v8 + 1006 > v11 )
          a1[2] = (void *)(v8 + 1006);
        else
          a1[2] = (void *)v11;
        v12 = realloc(v10);
        *a1 = (void *)v12;
        v10 = (char *)v12;
        if ( v12 )
        {
          v8 = (size_t)a1[1];
          goto LABEL_12;
        }
LABEL_76:
        abort();
      }
      goto LABEL_38;
    }
LABEL_15:
    v14 = *a2;
    if ( a3 <= v7 )
      v14 = a3;
    if ( !v14 )
    {
LABEL_18:
      result = (char *)(v7 - a3);
      a2[1] = a3 + v6;
      *a2 = (unsigned __int64)result;
      return result;
    }
    v15 = (char *)a1[1];
    v16 = (unsigned __int64)a1[2];
    v17 = *a1;
LABEL_20:
    v18 = (__int64)v17;
    if ( (unsigned __int64)&v15[v14] > v16 )
    {
      v19 = (unsigned __int64)&v15[v14 + 992];
      v20 = 2 * v16;
      if ( v19 > v20 )
        a1[2] = (void *)v19;
      else
        a1[2] = (void *)v20;
      v18 = realloc(v17);
      *a1 = (void *)v18;
      if ( !v18 )
        goto LABEL_76;
      v15 = (char *)a1[1];
    }
    memcpy(&v15[v18], (const void *)v6, v14);
    a1[1] = (char *)a1[1] + v14;
    v6 = a2[1];
    v7 = *a2;
    goto LABEL_18;
  }
  if ( v7 <= 6 )
    goto LABEL_15;
  if ( *(_DWORD *)v6 != 1852399455 || *(_WORD *)(v6 + 4) != 29801 || (v21 = 0, *(_BYTE *)(v6 + 6) != 90) )
    v21 = 1;
  v15 = (char *)a1[1];
  v17 = *a1;
  v16 = (unsigned __int64)a1[2];
  v22 = (size_t)a1[1];
  v23 = (char *)*a1;
  if ( v21 )
  {
    if ( *(_DWORD *)v6 != 1953914719 || *(_WORD *)(v6 + 4) != 27746 || *(_BYTE *)(v6 + 6) != 90 )
      goto LABEL_39;
    if ( (unsigned __int64)(v15 + 11) > v16 )
    {
      v31 = 2 * v16;
      if ( (unsigned __int64)(v15 + 1003) > v31 )
        a1[2] = v15 + 1003;
      else
        a1[2] = (void *)v31;
      v32 = realloc(v17);
      *a1 = (void *)v32;
      v23 = (char *)v32;
      if ( !v32 )
        goto LABEL_76;
      v22 = (size_t)a1[1];
    }
    memmove(v23 + 11, v23, v22);
    result = (char *)*a1;
    qmemcpy(*a1, "vtable for ", 11);
    a1[1] = (char *)a1[1] + 10;
    a2[1] += 6LL;
    *a2 -= 6LL;
  }
  else
  {
    if ( (unsigned __int64)(v15 + 16) > v16 )
    {
      v24 = 2 * v16;
      if ( (unsigned __int64)(v15 + 1008) > v24 )
        a1[2] = v15 + 1008;
      else
        a1[2] = (void *)v24;
      v25 = realloc(v17);
      *a1 = (void *)v25;
      v23 = (char *)v25;
      if ( !v25 )
        goto LABEL_76;
      v22 = (size_t)a1[1];
    }
    memmove(v23 + 16, v23, v22);
    result = (char *)*a1;
    *(__m128i *)*a1 = _mm_load_si128((const __m128i *)&xmmword_3F7B790);
    a1[1] = (char *)a1[1] + 15;
    a2[1] += 6LL;
    *a2 -= 6LL;
  }
  return result;
}
