// Function: sub_B12940
// Address: 0xb12940
//
__int64 __fastcall sub_B12940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v14; // [rsp+8h] [rbp-38h]

  v8 = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v14 = a6;
    v11 = sub_B91C10(a1, 38);
    a6 = v14;
    v8 = v11;
  }
  v12 = sub_B128C0(a2, a3, a4, v8, a5, a6, a7);
  sub_AA8740(*(_QWORD *)(a1 + 40), v12, a1);
  return v12;
}
