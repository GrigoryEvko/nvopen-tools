// Function: sub_1309DD0
// Address: 0x1309dd0
//
__int64 __fastcall sub_1309DD0(__int64 a1, _QWORD *a2, unsigned __int64 a3, unsigned __int64 a4, unsigned __int8 a5)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r13
  unsigned int v8; // r8d
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // r11
  __int64 v13; // r13
  int v14; // r9d
  bool v15; // sf
  char v16; // al
  __int64 v17; // r8
  unsigned __int8 v18; // r8
  __int64 v19; // r9
  char v20; // cl
  __int64 v21; // rax
  unsigned int v22; // [rsp+Ch] [rbp-44h]
  unsigned __int8 v23; // [rsp+Ch] [rbp-44h]
  _BYTE v24[49]; // [rsp+1Fh] [rbp-31h] BYREF

  v6 = a4;
  v7 = qword_505FA40[(unsigned __int8)(*a2 >> 20)];
  if ( a4 <= v7 )
    goto LABEL_2;
  v22 = a5;
  v16 = sub_1309680(a1, a2, a4, a5);
  a4 = v22;
  if ( !v16 )
  {
    if ( a1 )
    {
LABEL_13:
      v15 = --*(_DWORD *)(a1 + 152) < 0;
      if ( v15 && (unsigned __int8)sub_1309470((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112)) )
      {
        v23 = v18;
        sub_1315160(a1, v19, 0, 0);
        return v23;
      }
    }
    return 0;
  }
  if ( v6 <= a3 )
  {
LABEL_2:
    if ( a3 > v7 )
      goto LABEL_4;
LABEL_3:
    if ( v6 < v7 )
      goto LABEL_4;
    goto LABEL_19;
  }
  if ( a3 <= v7 )
    goto LABEL_3;
  if ( (unsigned __int8)sub_1309680(a1, a2, a3, v22) )
  {
LABEL_19:
    if ( a1 )
    {
      v15 = --*(_DWORD *)(a1 + 152) < 0;
      if ( v15 )
      {
        if ( (unsigned __int8)sub_1309470((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112)) )
          sub_1315160(a1, v17, 0, 0);
      }
    }
    return 0;
  }
LABEL_4:
  v8 = 1;
  if ( v6 < v7 )
  {
    v10 = qword_50579C0[*a2 & 0xFFFLL];
    v11 = sub_1316370(v10, qword_50579C0, a3, a4, 1);
    v12 = a2[2] & 0xFFFFFFFFFFFFF000LL;
    v13 = qword_505FA40[(unsigned __int8)(*a2 >> 20)];
    if ( !*(_QWORD *)(*(_QWORD *)(v11 + 8) + 56LL) )
      return 1;
    v24[0] = 0;
    if ( v6 > 0x1000 )
    {
      if ( v6 > 0x7000000000000000LL )
      {
        v14 = 232;
      }
      else
      {
        v20 = 7;
        _BitScanReverse64((unsigned __int64 *)&v21, 2 * v6 - 1);
        if ( (unsigned int)v21 >= 7 )
          v20 = v21;
        if ( (unsigned int)v21 < 6 )
          LODWORD(v21) = 6;
        v14 = ((((v6 - 1) & (-1LL << (v20 - 3))) >> (v20 - 3)) & 3) + 4 * v21 - 23;
      }
    }
    else
    {
      v14 = byte_5060800[(v6 + 7) >> 3];
    }
    if ( (unsigned __int8)sub_130B730(a1, (int)v10 + 10648, (_DWORD)a2, v12, dword_50607C0 + (int)v6, v14, (__int64)v24) )
      return 1;
    if ( v24[0] )
      sub_1314D40(a1, v10);
    sub_1314EA0(a1, v10, a2, v13);
    if ( !a1 )
      return 0;
    goto LABEL_13;
  }
  return v8;
}
