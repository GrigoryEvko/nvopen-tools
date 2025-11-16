// Function: sub_1DDC5F0
// Address: 0x1ddc5f0
//
__int64 __fastcall sub_1DDC5F0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 232);
  result = 0;
  if ( v1 )
    return *(_QWORD *)(*(_QWORD *)(v1 + 8) + 16LL);
  return result;
}
