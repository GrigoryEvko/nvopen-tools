// Function: sub_5CF860
// Address: 0x5cf860
//
__int64 *__fastcall sub_5CF860(char a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v3; // rdx

  for ( result = *(__int64 **)(a2 + 104); result; result = (__int64 *)*result )
  {
    if ( *((_BYTE *)result + 8) == a1 )
    {
      v3 = result[4];
      if ( v3 )
      {
        if ( !*(_QWORD *)v3 && *(_BYTE *)(v3 + 10) == 3 && *(_BYTE *)(*(_QWORD *)(v3 + 40) + 173LL) == 2 )
          break;
      }
    }
  }
  return result;
}
