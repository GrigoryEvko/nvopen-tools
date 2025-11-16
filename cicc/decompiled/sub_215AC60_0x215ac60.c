// Function: sub_215AC60
// Address: 0x215ac60
//
__int64 __fastcall sub_215AC60(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi
  _QWORD v4[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v5; // [rsp+10h] [rbp-30h] BYREF
  __int16 v6; // [rsp+20h] [rbp-20h]

  sub_3970E40();
  result = sub_214D850(a1, a2);
  if ( (_BYTE)result )
  {
    v3 = *(_QWORD *)(a1 + 256);
    v4[1] = 21;
    v4[0] = "\t.pragma \"nounroll\";\n";
    v6 = 261;
    v5 = v4;
    return sub_38DD5A0(v3, &v5);
  }
  return result;
}
