// Function: sub_134CBB0
// Address: 0x134cbb0
//
__int64 __fastcall sub_134CBB0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  _QWORD *v3; // r13
  unsigned int v4; // ebx
  _QWORD *v5; // r14
  __int64 result; // rax

  v3 = *(_QWORD **)(a1 + 56);
  v4 = a3;
  v5 = *(_QWORD **)(a1 + 48);
  if ( v5 == v3 )
    return 0;
  while ( 1 )
  {
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)*v5 + 32LL))(*v5, a2, v4);
    if ( (_BYTE)result )
      break;
    if ( v3 == ++v5 )
      return 0;
  }
  return result;
}
