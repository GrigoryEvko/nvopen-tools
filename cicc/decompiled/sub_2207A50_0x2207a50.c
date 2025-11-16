// Function: sub_2207A50
// Address: 0x2207a50
//
__int64 __fastcall sub_2207A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  result = sub_22522E0();
  if ( !(_BYTE)result )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 16) + 48LL))(
             *(_QWORD *)(a1 + 16),
             a2,
             a3,
             a4);
  return result;
}
