// Function: sub_302DA90
// Address: 0x302da90
//
__int64 __fastcall sub_302DA90(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // rdi
  _QWORD v4[4]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v5; // [rsp+20h] [rbp-20h]

  sub_31E1760();
  result = sub_3022840(a1, a2);
  if ( (_BYTE)result )
  {
    v3 = *(__int64 **)(a1 + 224);
    v4[1] = 21;
    v5 = 261;
    v4[0] = "\t.pragma \"nounroll\";\n";
    return sub_E99A90(v3, (__int64)v4);
  }
  return result;
}
