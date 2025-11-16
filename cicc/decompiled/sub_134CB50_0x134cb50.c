// Function: sub_134CB50
// Address: 0x134cb50
//
__int64 __fastcall sub_134CB50(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r14
  __int64 result; // rax

  v3 = *(_QWORD **)(a1 + 48);
  v4 = *(_QWORD **)(a1 + 56);
  if ( v3 == v4 )
    return 1;
  while ( 1 )
  {
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v3 + 24LL))(*v3, a2, a3);
    if ( (_BYTE)result != 1 )
      break;
    if ( v4 == ++v3 )
      return 1;
  }
  return result;
}
