// Function: sub_371BA10
// Address: 0x371ba10
//
__int64 __fastcall sub_371BA10(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v7)(__int64 *, __int64 *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v8)(); // [rsp+18h] [rbp-28h]

  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 136) = a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0x600000000LL;
  *(_QWORD *)(a1 + 120) = a2;
  *(_QWORD *)(a1 + 128) = a1;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  v3 = *a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v6[0] = sub_B9B140(v3, "sandboxregion", 0xDu);
  v4 = sub_B9C770(v3, v6, (__int64 *)1, 1, 1);
  v6[0] = a1;
  *(_QWORD *)(a1 + 112) = v4;
  v8 = sub_371C6C0;
  v7 = sub_371B6B0;
  *(_QWORD *)(a1 + 176) = sub_318AEA0((__int64)a2, (__int64)v6);
  if ( v7 )
    v7(v6, v6, 3);
  v6[0] = a1;
  v8 = sub_371C1D0;
  v7 = sub_371B6E0;
  *(_QWORD *)(a1 + 184) = sub_318ADE0((__int64)a2, (__int64)v6);
  result = (__int64)v7;
  if ( v7 )
    return v7(v6, v6, 3);
  return result;
}
