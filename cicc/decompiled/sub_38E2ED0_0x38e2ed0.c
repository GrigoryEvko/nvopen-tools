// Function: sub_38E2ED0
// Address: 0x38e2ed0
//
__int64 __fastcall sub_38E2ED0(__int64 a1, int a2)
{
  __int64 result; // rax
  int v3; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-40h] BYREF
  char *v5; // [rsp+20h] [rbp-30h] BYREF
  char v6; // [rsp+30h] [rbp-20h]
  char v7; // [rsp+31h] [rbp-1Fh]

  v3 = a2;
  v4[0] = a1;
  v4[1] = &v3;
  result = sub_3909F10(a1, sub_38F2290, v4, 1);
  if ( (_BYTE)result )
  {
    v7 = 1;
    v5 = " in directive";
    v6 = 3;
    return sub_39094A0(a1, &v5);
  }
  return result;
}
