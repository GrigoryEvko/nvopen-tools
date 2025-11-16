// Function: sub_2E6DEF0
// Address: 0x2e6def0
//
__int64 __fastcall sub_2E6DEF0(__int64 *a1, __int64 a2, __int64 a3)
{
  void *v4; // rax
  void *v5; // rax
  void *v6; // rax
  void *v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdi
  void *v12; // rax
  _QWORD *v13; // rdi
  _BYTE *v14; // rax
  __int64 *v15; // rdi
  __int64 result; // rax

  v4 = sub_CB72A0();
  sub_904010((__int64)v4, "Incorrect DFS numbers for:\n\tParent ");
  sub_2E6C8E0(*a1);
  v5 = sub_CB72A0();
  sub_904010((__int64)v5, "\n\tChild ");
  sub_2E6C8E0(a2);
  if ( a3 )
  {
    v6 = sub_CB72A0();
    sub_904010((__int64)v6, "\n\tSecond child ");
    sub_2E6C8E0(a3);
  }
  v7 = sub_CB72A0();
  sub_904010((__int64)v7, "\nAll children: ");
  v8 = a1[1];
  v9 = *(__int64 **)v8;
  v10 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
  if ( v10 != *(_QWORD *)v8 )
  {
    do
    {
      v11 = *v9++;
      sub_2E6C8E0(v11);
      v12 = sub_CB72A0();
      sub_904010((__int64)v12, ", ");
    }
    while ( (__int64 *)v10 != v9 );
  }
  v13 = sub_CB72A0();
  v14 = (_BYTE *)v13[4];
  if ( (unsigned __int64)v14 >= v13[3] )
  {
    sub_CB5D20((__int64)v13, 10);
  }
  else
  {
    v13[4] = v14 + 1;
    *v14 = 10;
  }
  v15 = (__int64 *)sub_CB72A0();
  result = v15[2];
  if ( v15[4] != result )
    return sub_CB5AE0(v15);
  return result;
}
