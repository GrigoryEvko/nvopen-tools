// Function: sub_1B28160
// Address: 0x1b28160
//
__int64 __fastcall sub_1B28160(_QWORD **a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 result; // rax
  bool v12; // zf

  v8 = (__int64 *)sub_1643270(*a1);
  v9 = sub_1644EA0(v8, a4, a5, 0);
  v10 = sub_1632080((__int64)a1, a2, a3, v9, 0);
  result = sub_1B28080(v10);
  v12 = (*(_BYTE *)(result + 32) & 0x30) == 0;
  *(_BYTE *)(result + 32) &= 0xF0u;
  if ( !v12 )
    *(_BYTE *)(result + 33) |= 0x40u;
  return result;
}
