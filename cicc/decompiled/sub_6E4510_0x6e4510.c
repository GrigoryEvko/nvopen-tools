// Function: sub_6E4510
// Address: 0x6e4510
//
__int64 __fastcall sub_6E4510(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v6; // r15
  __int64 v7; // rax
  int v8; // r8d
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // [rsp-10h] [rbp-50h]
  int v16; // [rsp+Ch] [rbp-34h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v7 = sub_6E3DA0(v6, 0);
  v8 = *(_DWORD *)(a1 + 40);
  v9 = *(_QWORD *)(v6 + 56);
  v15 = *(_QWORD *)(a1 + 48);
  v10 = *(_QWORD *)(a1 + 32);
  v11 = *(_QWORD *)(a1 + 24);
  v16 = 0;
  v12 = v7;
  v13 = sub_8A4520(v9, v11, v10, (int)v7 + 68, v8 & 0x4140 | 4u, (unsigned int)&v16, v15);
  if ( v16 )
    *(_BYTE *)(a1 + 56) = 1;
  *a2 = v13;
  result = *(_QWORD *)(v12 + 68);
  *a3 = result;
  return result;
}
