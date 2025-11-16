// Function: sub_2730090
// Address: 0x2730090
//
__int64 __fastcall sub_2730090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // [rsp-10h] [rbp-50h]

  v6 = (__int64)(0xCF3CF3CF3CF3CF3DLL * ((a2 - a1) >> 3) + 1) / 2;
  v7 = 168 * v6;
  v8 = a1 + 168 * v6;
  if ( v6 <= a4 )
  {
    sub_272ECC0(a1, a1 + 168 * v6, a3, a4, a5);
    sub_272ECC0(v8, a2, a3, v10, v11);
  }
  else
  {
    sub_2730090(a1, a1 + 168 * v6, a3);
    sub_2730090(v8, a2, a3);
  }
  sub_272F660(a1, v8, a2, 0xCF3CF3CF3CF3CF3DLL * (v7 >> 3), 0xCF3CF3CF3CF3CF3DLL * ((a2 - v8) >> 3), a3, a4);
  return v12;
}
