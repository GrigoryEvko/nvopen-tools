// Function: sub_15E1540
// Address: 0x15e1540
//
__int64 __fastcall sub_15E1540(__int64 a1, unsigned __int64 a2)
{
  void ***v2; // rax
  void ***v3; // rbx
  void (**v4)(); // rax
  void (**v5)(); // r15
  unsigned int v6; // eax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  size_t v9; // r8
  void *v10; // r14
  char (**v11)[5]; // r15
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rcx
  char (**v15)[5]; // rbx
  size_t v16; // r13
  char *v17; // rdi
  __int64 *v18; // rbx
  __int64 v19; // r12
  int v20; // eax
  unsigned int v21; // r12d
  size_t v24; // [rsp+10h] [rbp-60h]
  size_t v25; // [rsp+18h] [rbp-58h]
  char v26; // [rsp+2Fh] [rbp-41h] BYREF
  void *s2; // [rsp+30h] [rbp-40h] BYREF
  size_t v28; // [rsp+38h] [rbp-38h]

  s2 = &v26;
  v2 = (void ***)(__readfsqword(0) - 24);
  *v2 = &s2;
  v3 = v2;
  v4 = (void (**)())(__readfsqword(0) - 32);
  *v4 = sub_15DE590;
  if ( !&_pthread_key_create )
  {
    v6 = -1;
    goto LABEL_35;
  }
  v5 = v4;
  v6 = pthread_once(&dword_4F9E14C, init_routine);
  if ( v6 || (s2 = &v26, *v3 = &s2, *v5 = sub_15DE590, (v6 = pthread_once(&dword_4F9E14C, init_routine)) != 0) )
LABEL_35:
    sub_4264C5(v6);
  if ( a2 <= 4 )
  {
    v8 = a2;
    v7 = 0;
  }
  else
  {
    v7 = a2 - 5;
    v8 = 5;
  }
  v28 = v7;
  s2 = (void *)(a1 + v8);
  v26 = 46;
  v9 = sub_16D20C0(&s2, &v26, 1, 0);
  if ( v9 == -1 )
  {
    v10 = s2;
    v9 = v28;
  }
  else
  {
    v10 = s2;
    if ( v9 && v9 > v28 )
      v9 = v28;
  }
  v11 = &off_4984C00;
  v12 = 14;
  do
  {
    while ( 1 )
    {
      v14 = v12 >> 1;
      v15 = &v11[4 * (v12 >> 1)];
      v16 = (size_t)v15[1];
      v17 = (char *)*v15;
      if ( v16 > v9 )
        break;
      if ( v16 )
      {
        v24 = v9;
        v13 = memcmp(v17, v10, (size_t)v15[1]);
        v14 = v12 >> 1;
        v9 = v24;
        if ( v13 )
          goto LABEL_17;
      }
      if ( v16 == v9 )
        goto LABEL_18;
LABEL_12:
      if ( v16 >= v9 )
        goto LABEL_18;
LABEL_13:
      v11 = v15 + 4;
      v12 = v12 - v14 - 1;
      if ( v12 <= 0 )
        goto LABEL_19;
    }
    if ( !v9 )
      goto LABEL_18;
    v25 = v9;
    v13 = memcmp(v17, v10, v9);
    v9 = v25;
    v14 = v12 >> 1;
    if ( !v13 )
      goto LABEL_12;
LABEL_17:
    if ( v13 < 0 )
      goto LABEL_13;
LABEL_18:
    v12 = v14;
  }
  while ( v14 > 0 );
LABEL_19:
  v18 = qword_4C6F388;
  v19 = 218;
  if ( v11 != (char (**)[5])&off_4984DC0 && v11[1] == (char (*)[5])v9 && (!v9 || !memcmp(*v11, v10, v9)) )
  {
    v19 = (__int64)v11[3];
    v18 = &qword_4C6F388[(_QWORD)v11[2]];
  }
  v20 = sub_1601B20(v18, v19, a1, a2);
  if ( v20 == -1 )
    return 0;
  v21 = v20 + (((char *)v18 - (char *)&off_4C6F380) >> 3);
  if ( strlen((const char *)v18[v20]) != a2 && !(unsigned __int8)sub_15E1520(v21) )
    return 0;
  return v21;
}
