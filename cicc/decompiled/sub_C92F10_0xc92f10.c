// Function: sub_C92F10
// Address: 0xc92f10
//
__int64 __fastcall sub_C92F10(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rcx
  char v6; // dl
  char v7; // al

  result = 0;
  if ( a3 <= a1[1] )
  {
    if ( a3 )
    {
      v5 = 0;
      while ( 1 )
      {
        v6 = *(_BYTE *)(*a1 + v5);
        if ( (unsigned __int8)(v6 - 65) < 0x1Au )
          v6 += 32;
        v7 = *(_BYTE *)(a2 + v5);
        if ( (unsigned __int8)(v7 - 65) < 0x1Au )
          v7 += 32;
        if ( v6 != v7 )
          break;
        if ( a3 == ++v5 )
          return 1;
      }
      return 0;
    }
    else
    {
      return 1;
    }
  }
  return result;
}
