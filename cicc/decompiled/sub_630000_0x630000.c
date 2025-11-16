// Function: sub_630000
// Address: 0x630000
//
__int64 __fastcall sub_630000(int a1, _BYTE *a2, int a3)
{
  int *v4; // rcx
  unsigned __int8 v5; // al
  __int64 v6; // rsi
  __int64 result; // rax
  int v8; // [rsp+Ch] [rbp-14h] BYREF

  v4 = &v8;
  v5 = a2[40];
  v8 = 0;
  if ( (v5 & 0x20) == 0 )
    v4 = 0;
  LODWORD(v6) = a1;
  if ( (a2[43] & 8) != 0 )
    v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 216) + 40LL) + 32LL);
  result = sub_87CD50(a1, v6, a3, 0, 1, 1, ((v5 >> 6) ^ 1) & 1, (__int64)v4);
  if ( v8 )
    a2[41] |= 2u;
  return result;
}
