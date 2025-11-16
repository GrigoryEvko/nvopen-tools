// Function: sub_D51F80
// Address: 0xd51f80
//
bool __fastcall sub_D51F80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 **a7,
        unsigned __int8 **a8,
        __int64 a9)
{
  __int64 v9; // r15
  __int64 v10; // rbx
  unsigned __int8 *v11; // r14
  unsigned __int8 *v12; // r13
  unsigned __int8 *v13; // r12
  bool v14; // zf
  unsigned __int8 v15; // al
  unsigned __int8 *v17; // [rsp+18h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 56) == a1 + 48 )
    return 1;
  v9 = a1 + 48;
  v10 = *(_QWORD *)(a1 + 56);
  while ( 1 )
  {
    v11 = 0;
    if ( v10 )
      v11 = (unsigned __int8 *)(v10 - 24);
    v12 = *a8;
    v17 = *a7;
    v13 = *(unsigned __int8 **)(a9 + 16);
    v14 = sub_991A70(v11, 0, 0, 0, 0, 1u, 0) == 0;
    v15 = *v11;
    if ( v14 && v15 != 84 && v15 != 31 )
      break;
    if ( (unsigned int)v15 - 42 <= 0x11 && v11 != v13 || (unsigned __int8)(v15 - 82) <= 1u && v11 != v12 && v11 != v17 )
      break;
    v10 = *(_QWORD *)(v10 + 8);
    if ( v9 == v10 )
      return 1;
  }
  return v9 == v10;
}
