// Function: sub_1497CC0
// Address: 0x1497cc0
//
__int64 __fastcall sub_1497CC0(__int64 a1, __int64 a2, int a3, __m128i a4, __m128i a5)
{
  __int64 *v6; // r13
  unsigned int v7; // r12d
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-68h]
  _BYTE v12[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v13; // [rsp+20h] [rbp-50h]
  char v14; // [rsp+30h] [rbp-40h]

  v6 = sub_1494E70(a1, a2, a4, a5);
  v7 = a3 & ~(unsigned int)sub_1479390((__int64)v6, *(_QWORD *)(a1 + 112));
  v8 = sub_145DF90(*(_QWORD *)(a1 + 112), (__int64)v6, v7);
  sub_1495190(a1, v8, a4, a5);
  v10 = a2;
  v11 = v7;
  sub_14974D0((__int64)v12, a1 + 32, (__int64)&v10);
  result = v13;
  if ( !v14 )
    *(_DWORD *)(v13 + 40) |= v7;
  return result;
}
