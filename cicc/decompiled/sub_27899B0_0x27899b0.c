// Function: sub_27899B0
// Address: 0x27899b0
//
__int16 __fastcall sub_27899B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v4; // bx
  __int64 v5; // rax
  __int16 result; // ax
  unsigned int v7; // esi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  char v11; // cl
  unsigned __int64 v12; // rax
  __int16 v13; // [rsp+Ch] [rbp-34h]
  unsigned __int64 *v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  v4 = *(_WORD *)(a1 + 2);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v15 = sub_AE2980(a3, *(_DWORD *)(v5 + 8) >> 8)[3];
  if ( v15 > 0x40 )
    sub_C43690((__int64)&v14, 0, 0);
  else
    v14 = 0;
  if ( (unsigned __int8)sub_B4DE60(a2, a3, (__int64)&v14) )
  {
    v7 = v15;
    v8 = (unsigned __int64)v14;
    if ( v15 > 0x40 )
      v8 = *v14;
    _BitScanReverse64(&v9, 1LL << (v4 >> 1));
    v10 = 0x8000000000000000LL >> ((unsigned __int8)v9 ^ 0x3Fu);
    v11 = -1;
    v12 = -(__int64)(v8 | v10) & (v8 | v10);
    if ( v12 )
    {
      _BitScanReverse64(&v12, v12);
      v11 = 63 - (v12 ^ 0x3F);
    }
    LOBYTE(result) = v11;
    HIBYTE(result) = 1;
  }
  else
  {
    result = 0;
    v7 = v15;
  }
  if ( v7 > 0x40 )
  {
    if ( v14 )
    {
      v13 = result;
      j_j___libc_free_0_0((unsigned __int64)v14);
      return v13;
    }
  }
  return result;
}
