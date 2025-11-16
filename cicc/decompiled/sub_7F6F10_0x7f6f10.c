// Function: sub_7F6F10
// Address: 0x7f6f10
//
__int64 __fastcall sub_7F6F10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = sub_8D7760(*(_QWORD *)(a1 + 16), a2, a3, a4, a5, a6);
  if ( (_DWORD)result )
    return (*(_BYTE *)a1 & 2) == 0
        || !(unsigned int)sub_8D4C80(*(_QWORD *)(***(_QWORD ***)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 152LL) + 168LL) + 8LL))
        || ****(_QWORD ****)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 152LL) + 168LL) != 0;
  return result;
}
