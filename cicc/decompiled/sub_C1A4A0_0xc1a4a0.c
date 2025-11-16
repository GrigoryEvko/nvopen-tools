// Function: sub_C1A4A0
// Address: 0xc1a4a0
//
__int64 __fastcall sub_C1A4A0(__int64 a1, __int16 a2, int a3, int a4, int a5, int a6)
{
  __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // [rsp-8h] [rbp-40h]
  _QWORD v13[5]; // [rsp+10h] [rbp-28h] BYREF

  v7 = a1 + 168;
  v13[0] = 0;
  v8 = sub_C55B80(v7, a1, a3, a4, a5, a6, (__int64)v13);
  if ( (_BYTE)v8 )
    return v8;
  v10 = v13[0];
  v11 = *(_QWORD *)(a1 + 192) == 0;
  *(_WORD *)(a1 + 14) = a2;
  *(_QWORD *)(a1 + 136) = v10;
  if ( v11 )
    sub_4263D6(v7, a1, v12);
  (*(void (__fastcall **)(__int64, _QWORD *, __int64))(a1 + 200))(a1 + 176, v13, v12);
  return v8;
}
