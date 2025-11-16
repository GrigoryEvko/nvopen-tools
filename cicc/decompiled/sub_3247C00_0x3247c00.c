// Function: sub_3247C00
// Address: 0x3247c00
//
__int64 __fastcall sub_3247C00(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD *))(*a1 + 96LL))(a1) && !(unsigned __int8)sub_321F6A0(a1[26], a2) )
    return 0;
  v2 = *a2;
  if ( (unsigned __int8)v2 <= 0x24u )
  {
    v3 = 0x140000F000LL;
    if ( _bittest64(&v3, v2) )
      return *(unsigned __int8 *)(a1[26] + 3691LL) ^ 1u;
  }
  if ( (_BYTE)v2 == 18 && (a2[36] & 8) == 0 )
    return *(unsigned __int8 *)(a1[26] + 3691LL) ^ 1u;
  else
    return 0;
}
