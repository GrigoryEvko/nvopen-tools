// Function: sub_12C6E90
// Address: 0x12c6e90
//
__int64 __fastcall sub_12C6E90(int a1, int a2, const char **a3, int *a4, _DWORD *a5)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  const char *v11; // rdi
  size_t v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  char **v17; // r13
  char *v18; // rax
  const char **v19; // r13
  size_t v20; // r10
  __int64 v21; // rax
  char **v22; // r15
  char *v23; // rax
  const char *v24; // r14
  char v25; // al
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  char *srca; // [rsp+18h] [rbp-48h]
  char src; // [rsp+18h] [rbp-48h]
  const char *v32; // [rsp+20h] [rbp-40h] BYREF
  size_t v33; // [rsp+28h] [rbp-38h]

  v9 = 8LL * a2;
  *a4 = 0;
  if ( (unsigned __int64)a2 > 0xFFFFFFFFFFFFFFFLL )
    v9 = -1;
  v10 = sub_2207820(v9);
  *a5 = 0;
  v11 = *a3;
  v28 = v10;
  v12 = 0;
  v32 = v11;
  if ( v11 )
    v12 = strlen(v11);
  v33 = v12;
  v13 = sub_16D24E0(&v32, " -", 2, 0);
  v14 = v33;
  v15 = 1;
  if ( v13 < v33 )
  {
    v15 = v33 + 1 - v13;
    v14 = v13;
  }
  v16 = *a4;
  srca = (char *)&v32[v14];
  *a4 = v16 + 1;
  v17 = (char **)(v28 + 8 * v16);
  v18 = (char *)sub_2207820(v15);
  *v17 = strcpy(v18, srca);
  if ( a2 > 1 )
  {
    src = 0;
    v19 = a3 + 1;
    v29 = (__int64)&a3[(unsigned int)(a2 - 2) + 2];
    do
    {
      if ( !(unsigned int)sub_12C6910(*v19, 0, a5) )
      {
        v24 = *v19;
        if ( **v19 == 45 && v24[1] == 103 && !v24[2] || !strcmp(*v19, "-debug-compile") )
        {
          src = 1;
        }
        else
        {
          v25 = src;
          if ( !strcmp(*v19, "-generate-line-info") )
            v25 = 1;
          src = v25;
        }
        v20 = strlen(v24);
        v21 = *a4;
        v22 = (char **)(v28 + 8 * v21);
        *a4 = v21 + 1;
        v23 = (char *)sub_2207820(v20 + 1);
        *v22 = strcpy(v23, v24);
      }
      ++v19;
    }
    while ( (const char **)v29 != v19 );
    if ( src )
      *a5 |= 0x10u;
  }
  switch ( a1 )
  {
    case 2:
      *a5 |= 0xAu;
      break;
    case 3:
      *a5 |= 4u;
      break;
    case 1:
      *a5 |= 1u;
      break;
  }
  return v28;
}
