// Function: sub_EC2D90
// Address: 0xec2d90
//
__int64 __fastcall sub_EC2D90(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  __int64 v3; // [rsp+8h] [rbp-48h] BYREF
  char *v4; // [rsp+10h] [rbp-40h] BYREF
  char v5; // [rsp+30h] [rbp-20h]
  char v6; // [rsp+31h] [rbp-1Fh]

  v3 = a1;
  result = sub_ECE300(*(_QWORD *)(a1 + 8), sub_EC3A60, &v3, 1);
  if ( (_BYTE)result )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v6 = 1;
    v4 = " in directive";
    v5 = 3;
    return sub_ECD7F0(v2, &v4);
  }
  return result;
}
