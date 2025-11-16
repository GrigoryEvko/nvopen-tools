// Function: sub_C55290
// Address: 0xc55290
//
__int64 __fastcall sub_C55290(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 result; // rax
  unsigned int v9; // r12d
  bool v10; // zf
  __int64 v11; // [rsp-8h] [rbp-40h]
  _BYTE v12[33]; // [rsp+17h] [rbp-21h] BYREF

  v7 = a1 + 152;
  v12[0] = 0;
  LODWORD(result) = sub_C54F80(v7, a1, a3, a4, a5, a6, v12);
  if ( (_BYTE)result )
    return (unsigned int)result;
  v9 = v12[0];
  if ( v12[0] )
  {
    sub_C525B0(*(char **)(a1 + 136));
    exit(0);
  }
  v10 = *(_QWORD *)(a1 + 176) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v10 )
    sub_4263D6(v7, a1, v11);
  (*(void (__fastcall **)(__int64, _BYTE *, __int64))(a1 + 184))(a1 + 160, v12, v11);
  return v9;
}
