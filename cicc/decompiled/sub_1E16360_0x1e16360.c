// Function: sub_1E16360
// Address: 0x1e16360
//
__int64 __fastcall sub_1E16360(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  int i; // ecx
  _BYTE *v4; // rdx

  v1 = *(_QWORD *)(a1 + 16);
  result = *(unsigned __int16 *)(v1 + 2);
  if ( (*(_BYTE *)(v1 + 8) & 1) != 0 )
  {
    for ( i = *(_DWORD *)(a1 + 40); i != (_DWORD)result; result = (unsigned int)(result + 1) )
    {
      v4 = (_BYTE *)(*(_QWORD *)(a1 + 32) + 40LL * (unsigned int)result);
      if ( !*v4 && (v4[3] & 0x20) != 0 )
        break;
    }
  }
  return result;
}
