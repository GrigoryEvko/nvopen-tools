// Function: sub_2DD5CD0
// Address: 0x2dd5cd0
//
char *__fastcall sub_2DD5CD0(char *src, char *a2, char *a3, char *a4, _QWORD *a5, __int64 a6)
{
  char *v6; // r15
  char *v8; // r13
  __int64 v10; // rax
  char v11; // bl
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  char *v15; // r8
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+18h] [rbp-48h]

  v6 = a3;
  v8 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v18 = *(_QWORD *)v8;
      v20 = *(_QWORD *)(*(_QWORD *)v6 + 24LL);
      v11 = sub_AE5020(a6, v20);
      v12 = sub_9208B0(a6, v20);
      v19 = *(_QWORD *)(v18 + 24);
      v13 = ((1LL << v11) + ((unsigned __int64)(v12 + 7) >> 3) - 1) >> v11 << v11;
      LOBYTE(v20) = sub_AE5020(a6, v19);
      if ( ((1LL << v20) + ((unsigned __int64)(sub_9208B0(a6, v19) + 7) >> 3) - 1) >> v20 << v20 > v13 )
      {
        v10 = *(_QWORD *)v6;
        ++a5;
        v6 += 8;
        *(a5 - 1) = v10;
        if ( v8 == a2 )
          break;
      }
      else
      {
        v14 = *(_QWORD *)v8;
        ++a5;
        v8 += 8;
        *(a5 - 1) = v14;
        if ( v8 == a2 )
          break;
      }
    }
    while ( v6 != a4 );
  }
  if ( a2 != v8 )
    memmove(a5, v8, a2 - v8);
  v15 = (char *)a5 + a2 - v8;
  if ( a4 != v6 )
    v15 = (char *)memmove(v15, v6, a4 - v6);
  return &v15[a4 - v6];
}
