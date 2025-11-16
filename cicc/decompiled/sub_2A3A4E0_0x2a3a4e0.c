// Function: sub_2A3A4E0
// Address: 0x2a3a4e0
//
__int64 __fastcall sub_2A3A4E0(__int64 a1, const void *a2, size_t a3)
{
  __int64 **v3; // rbx
  _BYTE *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  __int64 v9; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-58h]
  int v11; // [rsp+1Ch] [rbp-54h]
  __int64 v12[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v13; // [rsp+40h] [rbp-30h]

  v3 = *(__int64 ***)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 72LL) + 40LL);
  v12[0] = sub_B9B140(*v3, a2, a3);
  v4 = (_BYTE *)sub_B9C770(*v3, v12, (__int64 *)1, 0, 1);
  v5 = sub_B9F6F0(*v3, v4);
  v6 = *(_QWORD *)(a1 + 72);
  v8 = v5;
  v13 = 257;
  v11 = 0;
  v9 = sub_AE4420((__int64)(v3 + 39), v6, 0);
  return sub_B33D10(a1, 0x12Du, (__int64)&v9, 1, (int)&v8, 1, v10, (__int64)v12);
}
