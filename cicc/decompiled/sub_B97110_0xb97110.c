// Function: sub_B97110
// Address: 0xb97110
//
void __fastcall sub_B97110(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx

  v4 = a1;
  v5 = *(_BYTE *)(a1 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_QWORD *)(a1 - 32);
  else
    v6 = a1 - 8LL * ((v5 >> 2) & 0xF) - 16;
  v7 = (__int64 *)(v6 + 8LL * a2);
  v8 = *v7;
  if ( (*(_BYTE *)(a1 + 1) & 0x7F) != 0 )
    v4 = 0;
  if ( v8 )
    sub_B91220((__int64)v7, v8);
  *v7 = a3;
  if ( a3 )
  {
    v9 = 1;
    if ( v4 )
      v9 = v4 & 0xFFFFFFFFFFFFFFFCLL | 1;
    sub_B96E90((__int64)v7, a3, v9);
  }
}
