// Function: sub_1633DB0
// Address: 0x1633db0
//
__int64 __fastcall sub_1633DB0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r8

  result = *a2;
  *a2 = 0;
  v3 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = result;
  if ( v3 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  return result;
}
