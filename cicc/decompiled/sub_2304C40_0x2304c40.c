// Function: sub_2304C40
// Address: 0x2304c40
//
__int64 *__fastcall sub_2304C40(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int16 v3; // bx
  __int64 v4; // rax

  v3 = sub_C09840(a2 + 8, a3);
  v4 = sub_22077B0(0x10u);
  if ( v4 )
  {
    *(_WORD *)(v4 + 8) = v3;
    *(_QWORD *)v4 = &unk_4A0B240;
  }
  *a1 = v4;
  return a1;
}
