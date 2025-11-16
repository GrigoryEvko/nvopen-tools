// Function: sub_2797D50
// Address: 0x2797d50
//
__int64 __fastcall sub_2797D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  unsigned int v11; // eax
  int v12; // eax
  _BYTE *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  const char *v20; // rdx
  const char *v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  int v24; // r13d
  __int64 v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  const char *v29[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v6 = a1 + 136;
  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v28 = 32LL * v7;
  v8 = 0;
  if ( v7 )
  {
    do
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v9 = *(_QWORD *)(a2 - 8);
      else
        v9 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v10 = *(_BYTE **)(v9 + v8);
      if ( *v10 > 0x16u )
      {
        v27 = *(_QWORD *)(v9 + v8);
        if ( !(unsigned __int8)sub_278A6A0(v6, (__int64)v10) )
          return 0;
        v11 = sub_278A710(v6, v27, 1);
        v12 = sub_2797350(v6, a3, a4, v11);
        v13 = sub_278BCD0(a1, a3, v12);
        if ( !v13 )
          return 0;
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v14 = *(_QWORD *)(a2 - 8);
        else
          v14 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v15 = v8 + v14;
        if ( *(_QWORD *)v15 )
        {
          v16 = *(_QWORD *)(v15 + 8);
          **(_QWORD **)(v15 + 16) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v15 + 16);
        }
        *(_QWORD *)v15 = v13;
        v17 = *((_QWORD *)v13 + 2);
        *(_QWORD *)(v15 + 8) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = v15 + 8;
        *(_QWORD *)(v15 + 16) = v13 + 16;
        *((_QWORD *)v13 + 2) = v15;
      }
      v8 += 32;
    }
    while ( v28 != v8 );
  }
  v18 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v18 == a3 + 48 )
  {
    v19 = 0;
  }
  else
  {
    if ( !v18 )
      BUG();
    v19 = v18 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 >= 0xB )
      v19 = 0;
  }
  sub_B44220((_QWORD *)a2, v19 + 24, 0);
  v29[0] = sub_BD5D20(a2);
  v30 = 773;
  v29[1] = v20;
  v29[2] = ".pre";
  sub_BD6B50((unsigned __int8 *)a2, v29);
  v21 = *(const char **)(a2 + 48);
  v29[0] = v21;
  if ( v21 )
  {
    sub_B96E90((__int64)v29, (__int64)v21, 1);
    v22 = *(_QWORD *)(a2 + 48);
    if ( v22 )
      sub_B91220(a2 + 48, v22);
    v23 = (unsigned __int8 *)v29[0];
    *(const char **)(a2 + 48) = v29[0];
    if ( v23 )
      sub_B976B0((__int64)v29, v23, a2 + 48);
  }
  sub_30EC360(*(_QWORD *)(a1 + 104), a2, a3);
  v24 = sub_2792F80(a1 + 136, a2);
  sub_2790CB0(a1 + 136, (_BYTE *)a2, v24);
  sub_27915B0(a1 + 352, v24, a2, a3);
  return 1;
}
