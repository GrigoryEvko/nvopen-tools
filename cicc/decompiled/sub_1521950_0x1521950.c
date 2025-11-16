// Function: sub_1521950
// Address: 0x1521950
//
__int64 __fastcall sub_1521950(__int64 a1, __int64 *a2, int a3)
{
  _BYTE *v3; // rax

  if ( !a3 )
    return sub_1519FE0(a1, 0);
  v3 = (_BYTE *)sub_15217C0(*a2, a3 - 1);
  return sub_1519FE0(a1, v3);
}
