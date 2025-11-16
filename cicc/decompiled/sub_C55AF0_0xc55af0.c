// Function: sub_C55AF0
// Address: 0xc55af0
//
__int64 __fastcall sub_C55AF0(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned int v8; // r12d
  int v10; // eax
  bool v11; // zf
  __int64 v12; // [rsp-8h] [rbp-40h]
  _DWORD v13[9]; // [rsp+14h] [rbp-24h] BYREF

  v7 = a1 + 160;
  v13[0] = 0;
  v8 = sub_C55A30(v7, a1, a3, a4, a5, a6, v13);
  if ( (_BYTE)v8 )
    return v8;
  v10 = v13[0];
  v11 = *(_QWORD *)(a1 + 184) == 0;
  *(_WORD *)(a1 + 14) = a2;
  *(_DWORD *)(a1 + 136) = v10;
  if ( v11 )
    sub_4263D6(v7, a1, v12);
  (*(void (__fastcall **)(__int64, _DWORD *, __int64))(a1 + 192))(a1 + 168, v13, v12);
  return v8;
}
