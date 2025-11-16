// Function: sub_EF7070
// Address: 0xef7070
//
__int64 __fastcall sub_EF7070(unsigned __int8 ***a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 **v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  char v12; // dl
  __int64 v13; // r12
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rsi
  __int64 result; // rax
  char v18; // [rsp+8h] [rbp-68h]
  int v20; // [rsp+1Ch] [rbp-54h] BYREF
  unsigned __int8 ***v21[10]; // [rsp+20h] [rbp-50h] BYREF

  v7 = *a1;
  v20 = a2;
  v21[0] = a1;
  *((_BYTE *)v7 + 937) = 1;
  v21[1] = (unsigned __int8 ***)&v20;
  v21[2] = (unsigned __int8 ***)(v7 + 101);
  v8 = sub_EF6F20(v21, a3, a4, a4, a5, a6);
  v18 = v12;
  if ( !v8 )
    return 2;
  v7[116] = (unsigned __int8 *)v8;
  v13 = v8;
  *((_BYTE *)v7 + 936) = 0;
  v14 = sub_EF6F20(v21, a5, a6, v9, v10, v11);
  v16 = v14;
  if ( !v14 )
    return 3;
  if ( v13 == v14 )
    return 0;
  if ( !v18 || *((_BYTE *)v7 + 936) )
  {
    result = 1;
    if ( v15 )
    {
      sub_EE6650((__int64)(v7 + 101), v16, v13);
      return 0;
    }
  }
  else
  {
    sub_EE6650((__int64)(v7 + 101), v13, v14);
    return 0;
  }
  return result;
}
