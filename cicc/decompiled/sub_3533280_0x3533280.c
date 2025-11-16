// Function: sub_3533280
// Address: 0x3533280
//
__int64 *__fastcall sub_3533280(char *a1, char *a2, char *a3, char *a4, __int64 *a5)
{
  char *v5; // r14
  char *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (__fastcall ***v9)(_QWORD); // r15
  int v10; // eax
  unsigned int v11; // r12d
  __int64 v12; // r15
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 *v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdi

  v5 = a3;
  v6 = a1;
  if ( a1 != a2 && a3 != a4 )
  {
    do
    {
      v9 = *(__int64 (__fastcall ****)(_QWORD))v5;
      v10 = (***(__int64 (__fastcall ****)(_QWORD))v5)(*(_QWORD *)v5);
      LODWORD(v9) = *((_DWORD *)v9 + 10);
      v11 = (_DWORD)v9 * (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)v6 + 8LL))(*(_QWORD *)v6) * v10;
      v12 = *(_QWORD *)v6;
      v13 = (***(__int64 (__fastcall ****)(_QWORD))v6)(*(_QWORD *)v6);
      LODWORD(v12) = *(_DWORD *)(v12 + 40);
      if ( v11 > (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)v5 + 8LL))(*(_QWORD *)v5)
               * v13
               * (unsigned int)v12 )
      {
        v7 = *(_QWORD *)v5;
        *(_QWORD *)v5 = 0;
        v8 = *a5;
        *a5 = v7;
        if ( v8 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 24LL))(v8);
        ++a5;
        v5 += 8;
        if ( v6 == a2 )
          break;
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
        if ( v6 == a2 )
          break;
      }
    }
    while ( v5 != a4 );
  }
  v16 = a2 - v6;
  v17 = (a2 - v6) >> 3;
  if ( a2 - v6 > 0 )
  {
    v18 = a5;
    do
    {
      v19 = *(_QWORD *)v6;
      *(_QWORD *)v6 = 0;
      v20 = *v18;
      *v18 = v19;
      if ( v20 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 24LL))(v20);
      v6 += 8;
      ++v18;
      --v17;
    }
    while ( v17 );
    v21 = 8;
    if ( v16 > 0 )
      v21 = v16;
    a5 = (__int64 *)((char *)a5 + v21);
  }
  v22 = a4 - v5;
  v23 = (a4 - v5) >> 3;
  if ( a4 - v5 > 0 )
  {
    v24 = a5;
    do
    {
      v25 = *(_QWORD *)v5;
      *(_QWORD *)v5 = 0;
      v26 = *v24;
      *v24 = v25;
      if ( v26 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 24LL))(v26);
      v5 += 8;
      ++v24;
      --v23;
    }
    while ( v23 );
    if ( v22 <= 0 )
      v22 = 8;
    return (__int64 *)((char *)a5 + v22);
  }
  return a5;
}
