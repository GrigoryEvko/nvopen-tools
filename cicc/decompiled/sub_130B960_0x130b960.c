// Function: sub_130B960
// Address: 0x130b960
//
__int64 __fastcall sub_130B960(__int64 a1, __int64 a2, _QWORD *a3, _BYTE *a4)
{
  __int64 v6; // r14
  __int64 result; // rax

  v6 = sub_131C0E0(*(_QWORD *)(a2 + 58376));
  if ( (*a3 & 0x10000) != 0 && (*a3 & 0x1000LL) == 0 )
    sub_130D990(a1, v6, a3, *(_QWORD *)(a2 + 58384), 1, 1);
  result = sub_13453B0(a1, a2, v6, a2 + 56, a3);
  *a4 = 1;
  return result;
}
