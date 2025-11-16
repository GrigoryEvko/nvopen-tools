// Function: sub_DDCB50
// Address: 0xddcb50
//
__int64 __fastcall sub_DDCB50(__int64 *a1, __int64 a2, _BYTE *a3, _BYTE *a4, __int64 a5)
{
  __int64 result; // rax

  result = sub_DC3A60((__int64)a1, a2, a3, a4);
  if ( !(_BYTE)result )
    return sub_DDC560(a1, *(_QWORD *)(a5 + 40), a2, (__int64)a3, (__int64)a4);
  return result;
}
