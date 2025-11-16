// Function: sub_35330A0
// Address: 0x35330a0
//
char *__fastcall sub_35330A0(char *a1, char *a2, char *a3, char *a4, __int64 *a5)
{
  char *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (__fastcall ***v9)(_QWORD); // r14
  int v10; // eax
  unsigned int v11; // r12d
  __int64 v12; // r14
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v23; // r14
  __int64 v24; // r12
  __int64 *v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax

  if ( a2 != a1 )
  {
    v6 = a1;
    while ( a4 != a3 )
    {
      v9 = *(__int64 (__fastcall ****)(_QWORD))a3;
      v10 = (***(__int64 (__fastcall ****)(_QWORD))a3)(*(_QWORD *)a3);
      LODWORD(v9) = *((_DWORD *)v9 + 10);
      v11 = (_DWORD)v9 * (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)v6 + 8LL))(*(_QWORD *)v6) * v10;
      v12 = *(_QWORD *)v6;
      v13 = (***(__int64 (__fastcall ****)(_QWORD))v6)(*(_QWORD *)v6);
      LODWORD(v12) = *(_DWORD *)(v12 + 40);
      if ( v11 > (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)a3 + 8LL))(*(_QWORD *)a3)
               * v13
               * (unsigned int)v12 )
      {
        v7 = *(_QWORD *)a3;
        *(_QWORD *)a3 = 0;
        v8 = *a5;
        *a5 = v7;
        if ( v8 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 24LL))(v8);
        ++a5;
        a3 += 8;
        if ( a2 == v6 )
          goto LABEL_11;
      }
      else
      {
        v14 = *(_QWORD *)v6;
        *(_QWORD *)v6 = 0;
        v15 = *a5;
        *a5 = v14;
        if ( v15 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 24LL))(v15);
        ++a5;
        v6 += 8;
        if ( a2 == v6 )
          goto LABEL_11;
      }
    }
    v23 = a2 - v6;
    v24 = (a2 - v6) >> 3;
    if ( a2 - v6 <= 0 )
      return (char *)a5;
    v25 = a5;
    do
    {
      v26 = *(_QWORD *)v6;
      *(_QWORD *)v6 = 0;
      v27 = *v25;
      *v25 = v26;
      if ( v27 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 24LL))(v27);
      v6 += 8;
      ++v25;
      --v24;
    }
    while ( v24 );
    v28 = 8;
    if ( v23 > 0 )
      v28 = v23;
    a5 = (__int64 *)((char *)a5 + v28);
  }
LABEL_11:
  v16 = a4 - a3;
  v17 = (a4 - a3) >> 3;
  if ( a4 - a3 <= 0 )
    return (char *)a5;
  v18 = a5;
  do
  {
    v19 = *(_QWORD *)a3;
    *(_QWORD *)a3 = 0;
    v20 = *v18;
    *v18 = v19;
    if ( v20 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 24LL))(v20);
    a3 += 8;
    ++v18;
    --v17;
  }
  while ( v17 );
  v21 = 8;
  if ( v16 > 0 )
    v21 = v16;
  return (char *)a5 + v21;
}
