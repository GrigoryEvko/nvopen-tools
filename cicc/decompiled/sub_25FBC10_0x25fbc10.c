// Function: sub_25FBC10
// Address: 0x25fbc10
//
__int64 __fastcall sub_25FBC10(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 result; // rax

  v3 = sub_AA5030(a3, 1);
  if ( v3 )
    v3 -= 24;
  result = sub_25FBA70(a1, a2, v3);
  if ( result )
    return *(_QWORD *)(result + 40);
  return result;
}
