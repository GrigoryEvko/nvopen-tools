// Function: sub_2D283A0
// Address: 0x2d283a0
//
unsigned __int64 __fastcall sub_2D283A0(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 result; // rax
  __int64 v3; // rdx

  v1 = *(_QWORD *)(a1 + 8);
  sub_B14250(*(_QWORD *)(a1 + 16));
  result = v1 | 4;
  if ( v1 == v3 )
    return **(_QWORD **)(a1 + 16) & 0xFFFFFFFFFFFFFFFBLL;
  return result;
}
