// Function: sub_2C0D550
// Address: 0x2c0d550
//
__int64 __fastcall sub_2C0D550(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // rdi
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // [rsp+10h] [rbp-70h]
  __int64 v13; // [rsp+18h] [rbp-68h]
  __int64 v14[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v15; // [rsp+40h] [rbp-40h]

  v7 = *(__int64 **)(a1 + 72);
  v8 = *(_QWORD *)(a2 + 8);
  v15 = 257;
  v9 = sub_ACD6D0(v7);
  BYTE4(v13) = *(_BYTE *)(v8 + 8) == 18;
  LODWORD(v13) = *(_DWORD *)(v8 + 32);
  v10 = sub_B37620((unsigned int **)a1, v13, v9, v14);
  BYTE4(v12) = 0;
  v14[0] = a2;
  v14[2] = a3;
  v14[1] = v10;
  return sub_B35180(a1, v8, 0xA4u, (__int64)v14, 3u, v12, a4);
}
