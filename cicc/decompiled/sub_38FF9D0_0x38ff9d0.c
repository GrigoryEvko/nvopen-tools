// Function: sub_38FF9D0
// Address: 0x38ff9d0
//
__int64 __fastcall sub_38FF9D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  __int64 v3; // [rsp+8h] [rbp-38h] BYREF
  char *v4; // [rsp+10h] [rbp-30h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  v3 = a1;
  result = sub_3909F10(*(_QWORD *)(a1 + 8), sub_3900610, &v3, 1);
  if ( (_BYTE)result )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v6 = 1;
    v4 = " in directive";
    v5 = 3;
    return sub_39094A0(v2, &v4);
  }
  return result;
}
