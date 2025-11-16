// Function: sub_379A100
// Address: 0x379a100
//
__int64 __fastcall sub_379A100(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r15
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v11; // r15
  unsigned int v12; // ebx
  __int64 v13; // rsi
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  int v16; // [rsp+18h] [rbp-38h]

  v4 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v9 = *(_QWORD *)(v4 + 48) + 16LL * v8;
  if ( *(_WORD *)v9 != v6 || *(_QWORD *)(v9 + 8) != v7 && !v6 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v12 = v6;
    v13 = *(_QWORD *)(a2 + 80);
    v15 = v13;
    if ( v13 )
    {
      v14 = v7;
      sub_B96E90((__int64)&v15, v13, 1);
      v7 = v14;
    }
    v16 = *(_DWORD *)(a2 + 72);
    v4 = (__int64)sub_33FAF80(v11, 215, (__int64)&v15, v12, v7, (unsigned int)&v15, a3);
    if ( v15 )
      sub_B91220((__int64)&v15, v15);
  }
  return v4;
}
