// Function: sub_2C1D1D0
// Address: 0x2c1d1d0
//
void __fastcall sub_2C1D1D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rax
  int v6; // esi
  unsigned __int8 *v7; // r14
  unsigned int v8; // [rsp+8h] [rbp-68h]
  __int64 v9[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v10; // [rsp+30h] [rbp-40h]

  v9[0] = *(_QWORD *)(a1 + 88);
  if ( v9[0] )
    sub_2AAAFA0(v9);
  sub_2BF1A90(a2, (__int64)v9);
  sub_9C6650(v9);
  v3 = *(_QWORD *)(a2 + 904);
  v4 = sub_BCE1B0(*(__int64 **)(a1 + 168), *(_QWORD *)(a2 + 8));
  v5 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  v6 = *(_DWORD *)(a1 + 160);
  v10 = 257;
  v7 = (unsigned __int8 *)sub_2C13D90(v3, v6, v5, v4, (__int64)v9, 0, v8);
  sub_2BF26E0(a2, a1 + 96, (__int64)v7, 0);
  sub_2BF08A0(a2, v7, *(_BYTE **)(a1 + 136));
  if ( *v7 > 0x1Cu )
    sub_2AAF930(a1, v7);
}
