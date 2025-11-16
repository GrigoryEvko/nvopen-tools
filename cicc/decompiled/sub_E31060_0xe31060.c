// Function: sub_E31060
// Address: 0xe31060
//
void __fastcall sub_E31060(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  size_t v13; // rbx
  size_t v14; // rax
  const void *v15; // r15
  char *v16; // r8
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rax

  v6 = a2[1];
  v7 = a2[2];
  v8 = (char *)*a2;
  if ( v6 + 11 > v7 )
  {
    v9 = v6 + 1003;
    v10 = 2 * v7;
    if ( v9 <= v10 )
      a2[2] = v10;
    else
      a2[2] = v9;
    v11 = realloc(v8);
    *a2 = v11;
    v8 = (char *)v11;
    if ( !v11 )
      goto LABEL_16;
    v6 = a2[1];
  }
  qmemcpy(&v8[v6], "operator \"\"", 11);
  v12 = a2[1] + 11;
  a2[1] = v12;
  v13 = *(_QWORD *)(a1 + 24);
  if ( v13 )
  {
    v14 = a2[2];
    v15 = *(const void **)(a1 + 32);
    v16 = (char *)*a2;
    if ( v13 + v12 <= v14 )
    {
LABEL_13:
      memcpy(&v16[v12], v15, v13);
      a2[1] += v13;
      goto LABEL_7;
    }
    v17 = v13 + v12 + 992;
    v18 = 2 * v14;
    if ( v17 > v18 )
      a2[2] = v17;
    else
      a2[2] = v18;
    v19 = realloc(v16);
    *a2 = v19;
    v16 = (char *)v19;
    if ( v19 )
    {
      v12 = a2[1];
      goto LABEL_13;
    }
LABEL_16:
    abort();
  }
LABEL_7:
  sub_E2EB40(a1, (__int64)a2, a3);
}
