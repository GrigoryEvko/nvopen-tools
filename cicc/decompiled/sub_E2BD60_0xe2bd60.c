// Function: sub_E2BD60
// Address: 0xe2bd60
//
void __fastcall sub_E2BD60(__int64 a1, void **a2, unsigned int a3)
{
  char *v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  char *v11; // rdi
  unsigned int v12; // [rsp+Ch] [rbp-14h]

  v5 = (char *)a2[1];
  v6 = (unsigned __int64)a2[2];
  v7 = (char *)*a2;
  if ( (unsigned __int64)(v5 + 9) > v6 )
  {
    v8 = (unsigned __int64)(v5 + 1001);
    v9 = 2 * v6;
    if ( v8 > v9 )
      a2[2] = (void *)v8;
    else
      a2[2] = (void *)v9;
    v12 = a3;
    v10 = realloc(v7);
    *a2 = (void *)v10;
    v7 = (char *)v10;
    if ( !v10 )
      abort();
    v5 = (char *)a2[1];
    a3 = v12;
  }
  v11 = &v7[(_QWORD)v5];
  *(_QWORD *)v11 = 0x3A5D6B6E7568745BLL;
  v11[8] = 32;
  a2[1] = (char *)a2[1] + 9;
  sub_E2B920(a1, a2, a3);
}
