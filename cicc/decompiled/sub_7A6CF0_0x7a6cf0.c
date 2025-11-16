// Function: sub_7A6CF0
// Address: 0x7a6cf0
//
__int64 __fastcall sub_7A6CF0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax

  if ( !dword_4F077BC )
    return sub_6851C0(0xD32u, (_DWORD *)(a2 + 72));
  if ( (_DWORD)qword_4F077B4 )
    return sub_6851C0(0xD32u, (_DWORD *)(a2 + 72));
  result = qword_4F077A8;
  if ( !qword_4F077A8 )
    return sub_6851C0(0xD32u, (_DWORD *)(a2 + 72));
  if ( qword_4F077A8 > 0xEA5Fu )
  {
    result = *a1;
    if ( *(_QWORD *)(*a1 + 160) )
      return sub_6851C0(0xD32u, (_DWORD *)(a2 + 72));
  }
  return result;
}
