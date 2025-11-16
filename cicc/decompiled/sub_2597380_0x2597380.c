// Function: sub_2597380
// Address: 0x2597380
//
__int64 __fastcall sub_2597380(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  bool v4; // zf
  char v5; // al
  bool v7; // [rsp+Dh] [rbp-23h] BYREF
  char v8; // [rsp+Eh] [rbp-22h] BYREF
  bool v9; // [rsp+Fh] [rbp-21h] BYREF

  v2 = sub_2587710(a1, a2);
  *(_QWORD *)(a1 + 112) = v3;
  v4 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = v2;
  if ( v4 )
    return 1;
  if ( !*(_QWORD *)(a1 + 104) )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  if ( (unsigned __int8)sub_2588040(a2, a1, (__int64 *)(a1 + 72), 0, &v7, 0, 0)
    && (unsigned __int8)sub_2596DB0(a2, a1, (__m128i *)(a1 + 72), 0, &v8, 0, 0)
    && (unsigned __int8)sub_252A800(a2, (__m128i *)(a1 + 72), a1, &v9) )
  {
    return 1;
  }
  v5 = *(_BYTE *)(a1 + 96);
  v4 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_BYTE *)(a1 + 97) = v5;
  if ( v4 )
    *(_BYTE *)(a1 + 112) = 1;
  return 0;
}
