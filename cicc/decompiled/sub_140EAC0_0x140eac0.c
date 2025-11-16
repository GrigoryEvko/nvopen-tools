// Function: sub_140EAC0
// Address: 0x140eac0
//
__int64 __fastcall sub_140EAC0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rbx
  __int64 v7; // rdi
  unsigned int v8; // r14d
  __int64 v9; // rdi
  unsigned int v10; // r14d
  bool v11; // al
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 result; // rax
  bool v15; // [rsp+Bh] [rbp-45h]
  unsigned __int16 v17; // [rsp+15h] [rbp-3Bh]
  unsigned __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
  v7 = a1[3 * (1 - v6)];
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
    v15 = *(_QWORD *)(v7 + 24) == 0;
  else
    v15 = v8 == (unsigned int)sub_16A57B0(v7 + 24);
  v17 = 0;
  if ( a4 )
    LOBYTE(v17) = v15 + 1;
  v9 = a1[3 * (2 - v6)];
  v10 = *(_DWORD *)(v9 + 32);
  if ( v10 <= 0x40 )
    v11 = *(_QWORD *)(v9 + 24) == 1;
  else
    v11 = v10 - 1 == (unsigned int)sub_16A57B0(v9 + 24);
  v12 = *a1;
  if ( (unsigned __int8)sub_140E950((_QWORD *)a1[-3 * v6], v18, a2, a3, (v11 << 16) | (unsigned int)v17) )
  {
    v13 = v18[0];
    if ( *(_DWORD *)(v12 + 8) > 0x3FFFu || v18[0] <= 0xFFFFFFFFFFFFFFFFLL >> (64 - BYTE1(*(_DWORD *)(v12 + 8))) )
      return sub_159C470(v12, v13, 0);
  }
  result = 0;
  if ( a4 )
  {
    v13 = -(__int64)v15;
    return sub_159C470(v12, v13, 0);
  }
  return result;
}
