// Function: sub_35C59C0
// Address: 0x35c59c0
//
__int64 __fastcall sub_35C59C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rdx
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // r11
  unsigned int v11; // ecx
  __int64 v12; // rsi
  int v13; // edi

  v5 = *(_QWORD **)(a1 + 88);
  v8 = v5[1] + 24LL * a2;
  result = *(_DWORD *)(v8 + 16) >> 12;
  v10 = v5[7] + 2 * result;
  v11 = *(_DWORD *)(v8 + 16) & 0xFFF;
  v12 = v5[8] + 16LL * *(unsigned __int16 *)(v8 + 20);
  if ( v10 )
  {
    result = 0;
    do
    {
      if ( a4 & *(_QWORD *)(v12 + 8 * result + 8) | a3 & *(_QWORD *)(v12 + 8 * result) )
        *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * (v11 >> 6)) |= 1LL << v11;
      v13 = *(__int16 *)(v10 + result);
      result += 2;
      v11 += v13;
    }
    while ( (_WORD)v13 );
  }
  return result;
}
