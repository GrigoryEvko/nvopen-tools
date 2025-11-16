// Function: sub_BAA730
// Address: 0xbaa730
//
__int64 __fastcall sub_BAA730(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r8

  result = *a2;
  *a2 = 0;
  v3 = *(_QWORD *)(a1 + 152);
  *(_QWORD *)(a1 + 152) = result;
  if ( v3 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  return result;
}
