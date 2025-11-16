// Function: sub_254E550
// Address: 0x254e550
//
__int64 __fastcall sub_254E550(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // [rsp+8h] [rbp-18h]

  if ( (unsigned __int8)((unsigned __int64)sub_250ED40(*(_QWORD *)(a2 + 208)) >> 32) )
  {
    v4 = sub_250ED40(*(_QWORD *)(a2 + 208));
    if ( !BYTE4(v4) )
      abort();
    v2 = sub_250D180((__int64 *)(a1 + 72), a2);
    if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
      v2 = **(_QWORD **)(v2 + 16);
    result = *(_DWORD *)(v2 + 8) >> 8;
    if ( (_DWORD)v4 != (_DWORD)result )
    {
      if ( *(_DWORD *)(a1 + 100) == -1 )
        *(_DWORD *)(a1 + 100) = result;
      result = *(unsigned __int8 *)(a1 + 97);
      *(_BYTE *)(a1 + 96) = result;
    }
  }
  else
  {
    result = *(unsigned __int8 *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
  }
  return result;
}
