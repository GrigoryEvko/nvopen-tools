// Function: sub_15E08E0
// Address: 0x15e08e0
//
unsigned __int64 __fastcall sub_15E08E0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // r8
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rdi
  _BYTE v10[16]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v11; // [rsp+10h] [rbp-40h]

  result = *(_QWORD *)(a1 + 96);
  if ( result )
  {
    if ( result > 0x333333333333333LL )
      sub_4261EA(a1, a2, 0x333333333333333LL);
    v4 = *(_QWORD *)(a1 + 24);
    result = sub_22077B0(40 * result);
    v5 = *(_QWORD *)(a1 + 96);
    *(_QWORD *)(a1 + 88) = result;
    if ( (_DWORD)v5 )
    {
      v6 = 0;
      while ( 1 )
      {
        v7 = v6 + 1;
        v8 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8 * (v6 + 1));
        v11 = 257;
        v9 = result + 40 * v6;
        if ( v9 )
          result = sub_15E0280(v9, v8, (__int64)v10, a1, v6);
        v6 = v7;
        if ( (unsigned int)v5 == v7 )
          break;
        result = *(_QWORD *)(a1 + 88);
      }
    }
  }
  *(_WORD *)(a1 + 18) &= ~1u;
  return result;
}
