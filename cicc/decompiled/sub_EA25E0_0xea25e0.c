// Function: sub_EA25E0
// Address: 0xea25e0
//
__int64 __fastcall sub_EA25E0(__int64 a1, char a2)
{
  bool v2; // zf
  __int64 result; // rax
  char v4; // [rsp+Ch] [rbp-24h] BYREF
  _QWORD v5[3]; // [rsp+10h] [rbp-20h] BYREF

  v2 = *(_BYTE *)(a1 + 869) == 0;
  v4 = a2;
  if ( !v2 || (result = sub_EA2540(a1), !(_BYTE)result) )
  {
    v5[0] = a1;
    v5[1] = &v4;
    return sub_ECE300(a1, sub_EADDD0, v5, 1);
  }
  return result;
}
