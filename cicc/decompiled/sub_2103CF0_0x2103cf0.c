// Function: sub_2103CF0
// Address: 0x2103cf0
//
__int64 __fastcall sub_2103CF0(_QWORD *a1, __int64 a2)
{
  unsigned __int16 *v3; // r12
  __int64 result; // rax
  unsigned __int16 *v5; // r9
  _QWORD *v6; // rax
  __int64 v7; // rdi
  int v8; // r10d
  __int64 v9; // rdx
  __int64 v10; // r11
  unsigned int v11; // esi
  __int16 v12; // cx
  _WORD *v13; // rsi
  unsigned __int16 *v14; // rdi
  unsigned __int16 v15; // cx
  _DWORD *v16; // rsi
  unsigned __int16 *v17; // rdx

  v3 = *(unsigned __int16 **)(a2 + 160);
  result = sub_1DD77D0(a2);
  if ( v3 != (unsigned __int16 *)result )
  {
    v5 = (unsigned __int16 *)result;
    do
    {
      v6 = (_QWORD *)*a1;
      if ( !*a1 )
        BUG();
      v7 = v6[7];
      v8 = *((_DWORD *)v5 + 1);
      v9 = v6[1];
      result = v6[8];
      v10 = v9 + 24LL * *v5;
      v11 = *(_DWORD *)(v10 + 16);
      v12 = (v11 & 0xF) * *v5;
      v13 = (_WORD *)(v7 + 2LL * (v11 >> 4));
      v14 = v13 + 1;
      v15 = *v13 + v12;
      v16 = (_DWORD *)(result + 4LL * *(unsigned __int16 *)(v10 + 20));
LABEL_9:
      v17 = v14;
      while ( v17 )
      {
        if ( !*v16 || (v8 & *v16) != 0 )
          *(_QWORD *)(a1[1] + 8LL * (v15 >> 6)) |= 1LL << v15;
        result = *v17;
        ++v16;
        ++v17;
        v14 = 0;
        v15 += result;
        if ( !(_WORD)result )
          goto LABEL_9;
      }
      v5 += 4;
    }
    while ( v3 != v5 );
  }
  return result;
}
