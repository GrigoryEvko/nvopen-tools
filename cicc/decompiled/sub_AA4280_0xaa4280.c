// Function: sub_AA4280
// Address: 0xaa4280
//
__int64 __fastcall sub_AA4280(__int64 a1, __int16 a2, int a3, int a4, int a5, int a6)
{
  __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v10; // rdx
  bool v11; // zf
  _BYTE v12[33]; // [rsp+17h] [rbp-21h] BYREF

  v7 = a1 + 160;
  v12[0] = 0;
  v8 = sub_C54F80(v7, a1, a3, a4, a5, a6, (__int64)v12);
  if ( (_BYTE)v8 )
    return v8;
  v10 = v12[0];
  **(_BYTE **)(a1 + 136) = v12[0];
  v11 = *(_QWORD *)(a1 + 184) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v11 )
    sub_4263D6(v7, a1, v10);
  (*(void (__fastcall **)(__int64, _BYTE *))(a1 + 192))(a1 + 168, v12);
  return v8;
}
