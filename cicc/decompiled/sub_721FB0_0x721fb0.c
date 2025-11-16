// Function: sub_721FB0
// Address: 0x721fb0
//
__int64 __fastcall sub_721FB0(unsigned __int8 *a1, _QWORD *a2, int a3)
{
  __int64 v4; // rax
  __int64 result; // rax

  sub_823800(a2);
  if ( !(a3 | sub_7215C0(a1)) )
    sub_721C20(a2, unk_4F076B0);
  sub_721C20(a2, (char *)a1);
  v4 = a2[2];
  if ( (unsigned __int64)(v4 + 1) > a2[1] )
  {
    sub_823810(a2);
    v4 = a2[2];
  }
  *(_BYTE *)(a2[4] + v4) = 0;
  result = a2[4];
  ++a2[2];
  return result;
}
