// Function: sub_15ABED0
// Address: 0x15abed0
//
__int64 __fastcall sub_15ABED0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r13
  int v6; // eax
  int v7; // r13d
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 j; // r15
  __int64 v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 i; // [rsp+18h] [rbp-68h]
  const char *v16; // [rsp+20h] [rbp-60h] BYREF
  int v17; // [rsp+28h] [rbp-58h]
  const char *v18; // [rsp+30h] [rbp-50h] BYREF
  int v19; // [rsp+38h] [rbp-48h]
  char v20; // [rsp+40h] [rbp-40h]
  char v21; // [rsp+41h] [rbp-3Fh]

  v21 = 1;
  v18 = "llvm.dbg.cu";
  v20 = 3;
  v5 = sub_1632310(a2, &v18);
  v6 = 0;
  if ( v5 )
    v6 = sub_161F520(v5, &v18, v3, v4);
  v19 = v6;
  v18 = (const char *)v5;
  sub_1632FD0(&v18);
  v16 = (const char *)v5;
  v17 = 0;
  sub_1632FD0(&v16);
  v7 = v19;
  v19 = v17;
  v18 = v16;
  while ( v19 != v7 )
  {
    v8 = sub_1632FB0(&v18);
    sub_15AB8A0(a1, v8);
    ++v19;
    sub_1632FD0(&v18);
  }
  result = a2 + 24;
  v14 = *(_QWORD *)(a2 + 32);
  if ( v14 != a2 + 24 )
  {
    do
    {
      v10 = v14 - 56;
      if ( !v14 )
        v10 = 0;
      v11 = sub_1626D20(v10);
      if ( v11 )
        sub_15ABAC0(a1, v11);
      for ( i = *(_QWORD *)(v10 + 80); v10 + 72 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        for ( j = *(_QWORD *)(i + 24); i + 16 != j; j = *(_QWORD *)(j + 8) )
        {
          v13 = j - 24;
          if ( !j )
            v13 = 0;
          sub_15ABE10(a1, a2, v13);
        }
      }
      result = *(_QWORD *)(v14 + 8);
      v14 = result;
    }
    while ( a2 + 24 != result );
  }
  return result;
}
