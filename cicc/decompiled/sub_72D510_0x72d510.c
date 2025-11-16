// Function: sub_72D510
// Address: 0x72d510
//
__int64 __fastcall sub_72D510(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  sub_724C70(a2, 6);
  *(_BYTE *)(a2 + 176) = 1;
  *(_QWORD *)(a2 + 184) = a1;
  result = sub_72D2E0(*(_QWORD **)(a1 + 120));
  *(_QWORD *)(a2 + 128) = result;
  if ( a3 )
    return sub_72A420((__int64 *)a1);
  return result;
}
