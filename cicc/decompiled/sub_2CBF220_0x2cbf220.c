// Function: sub_2CBF220
// Address: 0x2cbf220
//
__int64 __fastcall sub_2CBF220(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdi
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  unsigned int v13; // r8d

  v5 = *(_QWORD *)(a2 + 24);
  v6 = **a1;
  LOBYTE(a5) = v5 != v6;
  if ( *(_BYTE *)v5 <= 0x1Cu )
    return a5;
  a5 = 0;
  if ( v5 == v6 )
    return a5;
  v7 = a1[1];
  v8 = *(_QWORD *)(v5 + 40);
  v9 = *v7;
  a5 = *(unsigned __int8 *)(*v7 + 84);
  if ( (_BYTE)a5 )
  {
    v10 = *(_QWORD **)(v9 + 64);
    v11 = &v10[*(unsigned int *)(v9 + 76)];
    if ( v10 != v11 )
    {
      while ( v8 != *v10 )
      {
        if ( v11 == ++v10 )
          return a5;
      }
      return 0;
    }
    return a5;
  }
  LOBYTE(v13) = sub_C8CA60(v9 + 56, v8) == 0;
  return v13;
}
