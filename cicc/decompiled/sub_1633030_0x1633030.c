// Function: sub_1633030
// Address: 0x1633030
//
__int64 __fastcall sub_1633030(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 168) = a2;
  if ( v2 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  return result;
}
