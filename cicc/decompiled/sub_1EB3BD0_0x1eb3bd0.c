// Function: sub_1EB3BD0
// Address: 0x1eb3bd0
//
__int64 __fastcall sub_1EB3BD0(__int64 a1)
{
  __int64 v1; // r8
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 688) = 0;
  if ( v1 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 16LL))(v1);
  return result;
}
