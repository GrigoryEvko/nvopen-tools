// Function: sub_2554560
// Address: 0x2554560
//
__int64 __fastcall sub_2554560(__int64 a1, __m128i *a2, __int64 a3, char a4)
{
  unsigned int v4; // r12d
  unsigned __int8 *v6; // rbx
  char v7; // al
  __int64 *v8; // rax
  unsigned __int8 v9; // [rsp+Fh] [rbp-41h]
  _QWORD v10[7]; // [rsp+18h] [rbp-38h] BYREF

  LODWORD(v10[0]) = 40;
  v4 = sub_2516400(a1, a2, (__int64)v10, 1, a4, 40);
  if ( !(_BYTE)v4 )
  {
    v6 = (unsigned __int8 *)sub_250D070(a2);
    if ( (unsigned __int8)sub_2509800(a2) != 2 )
    {
      v7 = sub_98ED60(v6, 0, 0, 0, 0);
      if ( v7 )
      {
        v9 = v7;
        v8 = (__int64 *)sub_BD5C60((__int64)v6);
        v10[0] = sub_A778C0(v8, 40, 0);
        sub_2516380(a1, a2->m128i_i64, (__int64)v10, 1, 0);
        return v9;
      }
    }
  }
  return v4;
}
