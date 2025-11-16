// Function: sub_BA9760
// Address: 0xba9760
//
__int64 __fastcall sub_BA9760(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = a2;
  if ( v2 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  return result;
}
