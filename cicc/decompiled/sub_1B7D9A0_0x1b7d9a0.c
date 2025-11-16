// Function: sub_1B7D9A0
// Address: 0x1b7d9a0
//
__int64 __fastcall sub_1B7D9A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char **a5)
{
  __int64 i; // r14
  char *v8; // r8
  __int64 v9; // rbx
  char *v10; // rdi
  _QWORD *v11; // r12
  _QWORD *v12; // r10
  char *v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r14
  char *v16; // r15
  __int64 j; // r12
  _QWORD *v18; // rbx
  __int64 (__fastcall *v19)(char *, char *, _QWORD, __int64, __int64); // rax
  char v20; // al
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  char *v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  char *v28; // [rsp+18h] [rbp-48h]

  if ( a2 >= (a3 - 1) / 2 )
  {
    v9 = a2;
    v12 = (_QWORD *)(a1 + 8 * a2);
  }
  else
  {
    v27 = (a3 - 1) / 2;
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
      if ( v9 >= v27 )
        break;
    }
    v12 = v11;
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v9 )
  {
    v22 = 2 * v9 + 2;
    v23 = *(_QWORD *)(a1 + 8 * v22 - 8);
    v9 = v22 - 1;
    *v12 = v23;
    v12 = (_QWORD *)(a1 + 8 * v9);
  }
  v13 = *a5;
  v28 = a5[2];
  v26 = *a5;
  v14 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    v15 = v9;
    v16 = &a5[1][(_QWORD)a5[3]];
    v24 = (unsigned __int8)v13 & 1;
    for ( j = (v9 - 1) / 2; ; j = (j - 1) / 2 )
    {
      v18 = (_QWORD *)(a1 + 8 * j);
      v19 = (__int64 (__fastcall *)(char *, char *, _QWORD, __int64, __int64))v26;
      if ( v24 )
        v19 = *(__int64 (__fastcall **)(char *, char *, _QWORD, __int64, __int64))&v26[*(_QWORD *)v16 - 1];
      v20 = v19(v16, v28, *v18, a4, v14);
      v12 = (_QWORD *)(a1 + 8 * v15);
      if ( !v20 )
        break;
      v15 = j;
      *v12 = *v18;
      if ( a2 >= j )
      {
        v12 = (_QWORD *)(a1 + 8 * j);
        break;
      }
    }
  }
  *v12 = a4;
  return a4;
}
