// Function: sub_A7EE20
// Address: 0xa7ee20
//
__int64 __fastcall sub_A7EE20(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _BYTE *v6; // rax
  _BYTE v8[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v9; // [rsp+20h] [rbp-30h]

  if ( *a2 <= 0x15u && (unsigned __int8)sub_AD7930(a2) )
    return a3;
  v6 = sub_A7EC40(a1, (__int64)a2, *(_DWORD *)(*(_QWORD *)(a3 + 8) + 32LL));
  v9 = 257;
  return sub_B36550(a1, v6, a3, a4, v8, 0);
}
