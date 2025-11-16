// Function: sub_38E3760
// Address: 0x38e3760
//
__int64 __fastcall sub_38E3760(__int64 a1, char a2)
{
  bool v2; // zf
  __int64 result; // rax
  char v4; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-40h] BYREF
  char *v6; // [rsp+20h] [rbp-30h] BYREF
  char v7; // [rsp+30h] [rbp-20h]
  char v8; // [rsp+31h] [rbp-1Fh]

  v2 = *(_BYTE *)(a1 + 845) == 0;
  v4 = a2;
  if ( !v2 || (result = sub_38E36C0(a1), !(_BYTE)result) )
  {
    v5[0] = a1;
    v5[1] = &v4;
    result = sub_3909F10(a1, sub_38EC8A0, v5, 1);
    if ( (_BYTE)result )
    {
      v8 = 1;
      v6 = " in directive";
      v7 = 3;
      return sub_39094A0(a1, &v6);
    }
  }
  return result;
}
