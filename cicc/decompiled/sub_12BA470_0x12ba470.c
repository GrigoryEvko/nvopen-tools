// Function: sub_12BA470
// Address: 0x12ba470
//
__int64 __fastcall sub_12BA470(_DWORD *a1, _DWORD *a2)
{
  char v2; // r13
  __int64 v3; // r14

  v2 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v3 = qword_4F92D80;
    v2 = 1;
    sub_16C30C0(qword_4F92D80);
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v3 = qword_4F92D80;
  }
  if ( a1 )
    *a1 = 2;
  if ( a2 )
    *a2 = 0;
  if ( v2 )
    sub_16C30E0(v3);
  return 0;
}
