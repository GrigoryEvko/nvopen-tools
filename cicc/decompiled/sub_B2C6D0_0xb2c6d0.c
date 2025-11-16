// Function: sub_B2C6D0
// Address: 0xb2c6d0
//
unsigned __int64 __fastcall sub_B2C6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // rdi
  _BYTE v12[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v13; // [rsp+20h] [rbp-40h]

  result = *(_QWORD *)(a1 + 104);
  if ( result )
  {
    if ( result > 0x333333333333333LL )
      sub_4261EA(a1, a2, 0x333333333333333LL, a4);
    v6 = *(_QWORD *)(a1 + 24);
    result = sub_22077B0(40 * result);
    v7 = *(_QWORD *)(a1 + 104);
    *(_QWORD *)(a1 + 96) = result;
    if ( (_DWORD)v7 )
    {
      v8 = 0;
      while ( 1 )
      {
        v9 = v8 + 1;
        v10 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8 * (v8 + 1));
        v13 = 257;
        v11 = result + 40 * v8;
        if ( v11 )
          result = sub_B2BA90(v11, v10, (__int64)v12, a1, v8);
        v8 = v9;
        if ( (unsigned int)v7 == v9 )
          break;
        result = *(_QWORD *)(a1 + 96);
      }
    }
  }
  *(_WORD *)(a1 + 2) &= ~1u;
  return result;
}
