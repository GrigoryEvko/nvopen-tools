// Function: sub_D32210
// Address: 0xd32210
//
__int64 __fastcall sub_D32210(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v10; // rdx
  bool v11; // zf
  _DWORD v12[9]; // [rsp+14h] [rbp-24h] BYREF

  v7 = a1 + 160;
  v12[0] = 0;
  v8 = sub_C55A30(v7, a1, a3, a4, a5, a6, v12);
  if ( (_BYTE)v8 )
    return v8;
  v10 = v12[0];
  **(_DWORD **)(a1 + 136) = v12[0];
  v11 = *(_QWORD *)(a1 + 184) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v11 )
    sub_4263D6(v7, a1, v10);
  (*(void (__fastcall **)(__int64, _DWORD *))(a1 + 192))(a1 + 168, v12);
  return v8;
}
