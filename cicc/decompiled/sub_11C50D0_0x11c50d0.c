// Function: sub_11C50D0
// Address: 0x11c50d0
//
__int64 __fastcall sub_11C50D0(__int64 a1, int a2)
{
  char v2; // r8
  __int64 result; // rax
  __int64 *v4; // rax
  __int64 v5; // rax

  v2 = sub_B2D640(a1, a2, 89);
  result = 0;
  if ( !v2 )
  {
    v4 = (__int64 *)sub_B2BE50(a1);
    v5 = sub_A77AD0(v4, 0);
    sub_B2D410(a1, a2, v5);
    return 1;
  }
  return result;
}
