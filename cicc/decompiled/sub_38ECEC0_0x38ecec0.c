// Function: sub_38ECEC0
// Address: 0x38ecec0
//
__int64 __fastcall sub_38ECEC0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax

  *a2 = 0;
  result = sub_38ECD60(a1, a2, a3);
  if ( !(_BYTE)result )
    return sub_38EB510(a1, 1u, a2, (__int64)a3);
  return result;
}
