// Function: sub_C92E90
// Address: 0xc92e90
//
__int64 __fastcall sub_C92E90(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r9
  unsigned __int64 v5; // rbx
  __int64 v6; // rcx
  unsigned __int8 v7; // dl
  unsigned __int8 v8; // al
  bool v9; // cf
  __int64 result; // rax

  v3 = a3;
  v5 = a1[1];
  if ( v5 <= a3 )
    v3 = a1[1];
  if ( v3 )
  {
    v6 = 0;
    do
    {
      v7 = *(_BYTE *)(*a1 + v6);
      if ( (unsigned __int8)(v7 - 65) < 0x1Au )
        v7 += 32;
      v8 = *(_BYTE *)(a2 + v6);
      if ( (unsigned __int8)(v8 - 65) < 0x1Au )
        v8 += 32;
      v9 = v7 < v8;
      if ( v7 != v8 )
        return v9 ? -1 : 1;
    }
    while ( ++v6 != v3 );
  }
  result = 0;
  v9 = v5 < a3;
  if ( v5 != a3 )
    return v9 ? -1 : 1;
  return result;
}
