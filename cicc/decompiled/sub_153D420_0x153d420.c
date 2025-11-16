// Function: sub_153D420
// Address: 0x153d420
//
void __fastcall sub_153D420(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 v5; // r15
  __int64 v6; // rbx

  if ( (__int64)a2 - a1 <= 224 )
  {
    sub_153CD00(a1, a2, a3);
  }
  else
  {
    v4 = ((__int64)a2 - a1) >> 5;
    v5 = a1 + 16 * v4;
    v6 = (16 * v4) >> 4;
    sub_153D420(a1, v5);
    sub_153D420(v5, a2);
    sub_153D2A0(a1, v5, (__int64)a2, v6, ((__int64)a2 - v5) >> 4, a3);
  }
}
