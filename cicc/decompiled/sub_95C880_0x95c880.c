// Function: sub_95C880
// Address: 0x95c880
//
__int64 __fastcall sub_95C880(int a1, int a2, const char **a3, int *a4, _DWORD *a5)
{
  __int64 v8; // rdi
  __int64 v9; // rax
  const char *v10; // rdi
  size_t v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  const char *v15; // r13
  char *v16; // rax
  char *v17; // r8
  __int64 v18; // rax
  const char **v19; // r13
  char v20; // r14
  size_t v21; // rax
  char *v22; // rax
  char *v23; // r9
  __int64 v24; // rax
  const char *v25; // r15
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  const char *v30; // [rsp+20h] [rbp-40h] BYREF
  size_t v31; // [rsp+28h] [rbp-38h]

  v8 = 8LL * a2;
  *a4 = 0;
  if ( (unsigned __int64)a2 > 0xFFFFFFFFFFFFFFFLL )
    v8 = -1;
  v9 = sub_2207820(v8);
  *a5 = 0;
  v10 = *a3;
  v28 = v9;
  v11 = 0;
  v30 = v10;
  if ( v10 )
    v11 = strlen(v10);
  v31 = v11;
  v12 = sub_C935B0(&v30, " -", 2, 0);
  v13 = v31;
  v14 = 1;
  if ( v12 < v31 )
  {
    v13 = v12;
    v14 = v31 + 1 - v12;
  }
  v15 = &v30[v13];
  v16 = (char *)sub_2207820(v14);
  v17 = strcpy(v16, v15);
  v18 = *a4;
  *(_QWORD *)(v28 + 8 * v18) = v17;
  *a4 = v18 + 1;
  if ( a2 > 1 )
  {
    v19 = a3 + 1;
    v20 = 0;
    v29 = (__int64)&a3[(unsigned int)(a2 - 2) + 2];
    do
    {
      if ( !(unsigned int)sub_95C230(*v19, 0, a5) )
      {
        v25 = *v19;
        if ( **v19 == 45 && v25[1] == 103 && !v25[2] || !strcmp(*v19, "-debug-compile") )
        {
          v20 = 1;
        }
        else if ( !strcmp(*v19, "-generate-line-info") )
        {
          v20 = 1;
        }
        v21 = strlen(*v19);
        v22 = (char *)sub_2207820(v21 + 1);
        v23 = strcpy(v22, v25);
        v24 = *a4;
        *(_QWORD *)(v28 + 8 * v24) = v23;
        *a4 = v24 + 1;
      }
      ++v19;
    }
    while ( (const char **)v29 != v19 );
    if ( v20 )
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
