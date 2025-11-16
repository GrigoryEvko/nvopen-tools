// Function: sub_5C8410
// Address: 0x5c8410
//
__int64 __fastcall sub_5C8410(__int64 a1, __int64 a2, char a3)
{
  char v3; // al
  char *v4; // rax

  v3 = *(_BYTE *)(a1 + 10);
  if ( v3 != 23 && (a3 != 29 || v3 != 1) )
  {
    v4 = sub_5C79F0(a1);
    sub_6849F0(7, 1835, a1 + 56, v4);
  }
  return a2;
}
