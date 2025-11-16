// Function: sub_2AF7880
// Address: 0x2af7880
//
__int64 __fastcall sub_2AF7880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char **a5)
{
  __int64 i; // r14
  char *v8; // r8
  __int64 v9; // rbx
  char *v10; // rdi
  _QWORD *v11; // r12
  _QWORD *v12; // r10
  char *v13; // rcx
  __int64 v14; // r14
  char *v15; // r15
  __int64 v16; // r12
  _QWORD *v17; // rbx
  __int64 (__fastcall *v18)(char *, char *, _QWORD, __int64); // rax
  char v19; // al
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-58h]
  char *v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  char *v27; // [rsp+18h] [rbp-48h]

  if ( a2 >= (a3 - 1) / 2 )
  {
    v9 = a2;
    v12 = (_QWORD *)(a1 + 8 * a2);
  }
  else
  {
    v26 = (a3 - 1) / 2;
    for ( i = a2; ; i = v9 )
    {
      v8 = *a5;
      v9 = 2 * (i + 1);
      v10 = &a5[1][(_QWORD)a5[3]];
      v11 = (_QWORD *)(a1 + 16 * (i + 1));
      if ( ((unsigned __int8)*a5 & 1) != 0 )
        v8 = *(char **)&v8[*(_QWORD *)v10 - 1];
      if ( ((unsigned __int8 (__fastcall *)(char *, char *, _QWORD, _QWORD))v8)(v10, a5[2], *v11, *(v11 - 1)) )
      {
        --v9;
        v11 = (_QWORD *)(a1 + 8 * v9);
      }
      *(_QWORD *)(a1 + 8 * i) = *v11;
      if ( v9 >= v26 )
        break;
    }
    v12 = v11;
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v9 )
  {
    v21 = 2 * v9 + 2;
    v22 = *(_QWORD *)(a1 + 8 * v21 - 8);
    v9 = v21 - 1;
    *v12 = v22;
    v12 = (_QWORD *)(a1 + 8 * v9);
  }
  v13 = *a5;
  v27 = a5[2];
  v25 = *a5;
  v14 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    v15 = &a5[1][(_QWORD)a5[3]];
    v23 = (unsigned __int8)v13 & 1;
    v16 = v9;
    while ( 1 )
    {
      v17 = (_QWORD *)(a1 + 8 * v14);
      v18 = (__int64 (__fastcall *)(char *, char *, _QWORD, __int64))v25;
      if ( v23 )
        v18 = *(__int64 (__fastcall **)(char *, char *, _QWORD, __int64))&v25[*(_QWORD *)v15 - 1];
      v19 = v18(v15, v27, *v17, a4);
      v12 = (_QWORD *)(a1 + 8 * v16);
      if ( !v19 )
        break;
      v16 = v14;
      *v12 = *v17;
      if ( v14 <= a2 )
      {
        v12 = (_QWORD *)(a1 + 8 * v14);
        break;
      }
      v14 = (v14 - 1) / 2;
    }
  }
  *v12 = a4;
  return a4;
}
