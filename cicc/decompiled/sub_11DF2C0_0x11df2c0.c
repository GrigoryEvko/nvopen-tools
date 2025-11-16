// Function: sub_11DF2C0
// Address: 0x11df2c0
//
__int64 __fastcall sub_11DF2C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r14
  _QWORD *v9; // rax
  __int64 result; // rax
  unsigned int v11[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = *(_QWORD *)(a2 + 32 * (1 - v4));
  if ( *(_BYTE *)v6 != 17 )
  {
    sub_98B430(v5, 8u);
    return 0;
  }
  v7 = sub_98B430(*(_QWORD *)(a2 - 32 * v4), 8u);
  v8 = v7;
  if ( !v7 )
    return 0;
  v11[0] = 0;
  sub_11DA2E0(a2, v11, 1, v7);
  v9 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( (unsigned __int64)v9 + 1 < v8 )
    return 0;
  result = sub_11CA0D0(v5, a3, *(__int64 **)(a1 + 24));
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
