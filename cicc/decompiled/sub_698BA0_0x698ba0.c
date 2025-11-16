// Function: sub_698BA0
// Address: 0x698ba0
//
__int64 __fastcall sub_698BA0(__int64 a1, __m128i *a2, __int64 a3, unsigned int a4, int a5, __m128i **a6)
{
  __int64 v8; // r14
  char v9; // al
  unsigned int v10; // r8d
  __m128i *v12; // rsi
  __int64 v13; // r12
  __m128i *v14; // rdi
  int v15; // r8d
  __int64 v16; // rax
  unsigned int v19; // [rsp+18h] [rbp-3A8h]
  __int64 v20; // [rsp+28h] [rbp-398h] BYREF
  _BYTE v21[160]; // [rsp+30h] [rbp-390h] BYREF
  _BYTE v22[352]; // [rsp+D0h] [rbp-2F0h] BYREF
  __m128i v23[4]; // [rsp+230h] [rbp-190h] BYREF
  _DWORD v24[83]; // [rsp+274h] [rbp-14Ch] BYREF

  v8 = *(_QWORD *)(a1 + 120);
  if ( (unsigned int)sub_8D32E0(v8) )
    v8 = sub_8D46C0(v8);
  while ( 1 )
  {
    v9 = *(_BYTE *)(v8 + 140);
    if ( v9 != 12 )
      break;
    v8 = *(_QWORD *)(v8 + 160);
  }
  v10 = 0;
  if ( v9 )
  {
    sub_6E1E00(4, v21, 0, 0);
    sub_68ACF0(a1, (__int64)v22);
    v12 = 0;
    v13 = sub_6E3060(v22);
    v14 = a2;
    sub_702840(a2, (__int64)v23, (__int64)&v20);
    if ( a5 )
    {
      v12 = (__m128i *)v24;
      v14 = v23;
      sub_6980A0(v23, v24, a4, 0, 0, 0);
    }
    v15 = 0;
    if ( v20 )
    {
      v16 = sub_73D4C0(v23[0].m128i_i64[0], dword_4F077C4 == 2);
      v12 = v23;
      v14 = (__m128i *)sub_736020(v16, 0);
      *a6 = v14;
      sub_68BC10((__int64)v14, v23);
      v15 = 1;
    }
    v19 = v15;
    sub_6E2B30(v14, v12);
    sub_6E1990(v13);
    return v19;
  }
  return v10;
}
