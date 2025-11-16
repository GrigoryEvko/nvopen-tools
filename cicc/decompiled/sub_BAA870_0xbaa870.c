// Function: sub_BAA870
// Address: 0xbaa870
//
__int64 __fastcall sub_BAA870(__int64 a1)
{
  _BYTE *v1; // rax

  v1 = (_BYTE *)sub_BA91D0(a1, "stack-protector-guard", 0x15u);
  if ( !v1 || *v1 )
    return 0;
  else
    return sub_B91420((__int64)v1);
}
