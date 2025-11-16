// Function: sub_15F9F50
// Address: 0x15f9f50
//
__int64 __fastcall sub_15F9F50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  int v9; // r14d
  int v10; // ecx
  __int64 v11; // r15

  v3 = a1;
  if ( !a3 )
    return v3;
  v5 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned __int8)v5 <= 0xFu && (v7 = 35454, _bittest64(&v7, v5))
    || ((unsigned int)(v5 - 13) <= 1 || (_DWORD)v5 == 16) && (unsigned __int8)sub_16435F0(a1, 0) )
  {
    v8 = 1;
    v9 = 1;
    if ( a3 == 1 )
      return v3;
    while ( 1 )
    {
      v10 = *(unsigned __int8 *)(v3 + 8);
      if ( (_BYTE)v10 != 14 && (v10 != 13 && v10 != 16 || *(_BYTE *)(v3 + 8) == 15) )
        break;
      v11 = *(_QWORD *)(a2 + 8 * v8);
      if ( !(unsigned __int8)sub_1643DA0(v3, v11) )
        break;
      v3 = sub_1643D30(v3, v11);
      v8 = (unsigned int)++v9;
      if ( a3 == v9 )
        return v3;
    }
  }
  return 0;
}
