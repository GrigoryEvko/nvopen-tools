// Function: sub_21361E0
// Address: 0x21361e0
//
__int64 __fastcall sub_21361E0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r13
  const void **v10; // r15
  unsigned int v11; // r12d
  __int64 v12; // r12
  __int128 v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+10h] [rbp-50h] BYREF
  int v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  int v18; // [rsp+28h] [rbp-38h]

  v6 = *(unsigned __int64 **)(a2 + 32);
  DWORD2(v14) = 0;
  v16 = 0;
  *(_QWORD *)&v14 = 0;
  v7 = v6[1];
  v15 = 0;
  sub_20174B0(a1, *v6, v7, &v14, &v15);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v11 = **(unsigned __int8 **)(a2 + 40);
  v17 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v17, v8, 2);
  v18 = *(_DWORD *)(a2 + 64);
  v12 = sub_1D309E0(v9, 145, (__int64)&v17, v11, v10, 0, a3, a4, a5, v14);
  if ( v17 )
    sub_161E7C0((__int64)&v17, v17);
  return v12;
}
