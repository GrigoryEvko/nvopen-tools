// Function: sub_22ADAF0
// Address: 0x22adaf0
//
_DWORD *__fastcall sub_22ADAF0(
        unsigned int *a1,
        unsigned int *a2,
        unsigned int *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int *v6; // r15
  _DWORD *v8; // r12
  unsigned int *v9; // rbx
  char **v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdi
  char **v14; // rsi
  __int64 v15; // rdx
  _DWORD *v16; // r13
  char **v17; // r14
  __int64 v18; // rcx
  char **v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // r15
  unsigned __int64 v22; // r13
  _DWORD *v23; // r14
  char **v24; // rbx
  int v25; // eax
  char **v26; // rsi
  __int64 v27; // rdi
  __int64 v29; // [rsp+8h] [rbp-48h]
  unsigned __int64 v30; // [rsp+10h] [rbp-40h]

  v6 = a1;
  v8 = (_DWORD *)a5;
  v9 = a3;
  if ( a2 != a1 && (unsigned int *)a4 != a3 )
  {
    do
    {
      v11 = *v9;
      v12 = *v6;
      v13 = (__int64)(v8 + 2);
      if ( (unsigned int)v11 > (unsigned int)v12 )
      {
        *v8 = v11;
        v10 = (char **)(v9 + 2);
        v8 += 18;
        v9 += 18;
        sub_22AD4A0(v13, v10, v12, v11, a5, a6);
        if ( a2 == v6 )
          break;
      }
      else
      {
        *v8 = v12;
        v14 = (char **)(v6 + 2);
        v6 += 18;
        v8 += 18;
        sub_22AD4A0(v13, v14, v12, v11, a5, a6);
        if ( a2 == v6 )
          break;
      }
    }
    while ( (unsigned int *)a4 != v9 );
  }
  v15 = 0x8E38E38E38E38E39LL;
  v29 = (char *)a2 - (char *)v6;
  v30 = 0x8E38E38E38E38E39LL * (((char *)a2 - (char *)v6) >> 3);
  if ( (char *)a2 - (char *)v6 > 0 )
  {
    v16 = v8 + 2;
    v17 = (char **)(v6 + 2);
    do
    {
      v18 = *((unsigned int *)v17 - 2);
      v19 = v17;
      v20 = (__int64)v16;
      v17 += 9;
      v16 += 18;
      *(v16 - 20) = v18;
      sub_22AD4A0(v20, v19, v15, v18, a5, a6);
      --v30;
    }
    while ( v30 );
    v15 = v29;
    if ( v29 <= 0 )
      v15 = 72;
    v8 = (_DWORD *)((char *)v8 + v15);
  }
  v21 = a4 - (_QWORD)v9;
  v22 = 0x8E38E38E38E38E39LL * ((a4 - (__int64)v9) >> 3);
  if ( a4 - (__int64)v9 > 0 )
  {
    v23 = v8 + 2;
    v24 = (char **)(v9 + 2);
    do
    {
      v25 = *((_DWORD *)v24 - 2);
      v26 = v24;
      v27 = (__int64)v23;
      v24 += 9;
      v23 += 18;
      *(v23 - 20) = v25;
      sub_22AD4A0(v27, v26, v15, a4, a5, a6);
      --v22;
    }
    while ( v22 );
    if ( v21 <= 0 )
      v21 = 72;
    return (_DWORD *)((char *)v8 + v21);
  }
  return v8;
}
