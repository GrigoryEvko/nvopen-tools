// Function: sub_22116C0
// Address: 0x22116c0
//
__int64 __fastcall sub_22116C0(__int64 a1, int a2, int a3, unsigned __int8 a4, int a5, int a6, _QWORD *a7)
{
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v14[4]; // [rsp+10h] [rbp-60h] BYREF
  __int64 (__fastcall *v15)(); // [rsp+30h] [rbp-40h]

  v15 = 0;
  sub_2216B00(v14, a7);
  v10 = *(_QWORD *)(a1 + 16);
  v14[1] = *(_QWORD *)(*a7 - 24LL);
  v15 = sub_2211A20;
  v11 = sub_2222970(v10, a2, a3, a4, a5, a6, (_TBYTE)0LL, (__int64)v14);
  if ( v15 )
    ((void (__fastcall *)(__int64 *))v15)(v14);
  return v11;
}
