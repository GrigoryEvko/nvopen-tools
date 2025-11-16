// Function: sub_12AE010
// Address: 0x12ae010
//
__int64 __fastcall sub_12AE010(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int64 *v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // r13
  char *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v16; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-60h] BYREF
  _BYTE v18[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  v7 = *(unsigned __int64 **)(a4 + 16);
  v8 = v7[2];
  v9 = sub_127A030(*(_QWORD *)(a2 + 32) + 8LL, *v7, 0);
  v10 = *(_QWORD **)(a2 + 32);
  v16 = v9;
  v11 = sub_126A190(v10, a3, (__int64)&v16, 1u);
  v17[0] = sub_128F980(a2, (__int64)v7);
  v12 = sub_128F980(a2, v8);
  v13 = *(_QWORD *)(v11 + 24);
  v17[1] = v12;
  v19 = 257;
  v14 = sub_1285290((__int64 *)(a2 + 48), v13, v11, (int)v17, 2, (__int64)v18, 0);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v14;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
