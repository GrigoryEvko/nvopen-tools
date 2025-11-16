// Function: sub_24770C0
// Address: 0x24770c0
//
void __fastcall sub_24770C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r10
  _QWORD *v5; // rax
  unsigned __int8 v6; // bl
  int v7; // eax
  int v8; // ecx
  int v9; // edx
  int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int16 v20; // dx
  __int64 v21; // [rsp+8h] [rbp-F8h]
  _QWORD v22[4]; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v23; // [rsp+30h] [rbp-D0h]
  unsigned int *v24[2]; // [rsp+40h] [rbp-C0h] BYREF
  char v25; // [rsp+50h] [rbp-B0h] BYREF
  void *v26; // [rsp+C0h] [rbp-40h]

  sub_2468350((__int64)v24, (_QWORD *)a2);
  v5 = sub_2463540((__int64 *)a1, *(_QWORD *)(a2 + 8));
  v3 = *(_QWORD *)(a2 - 32);
  v4 = (__int64)v5;
  _BitScanReverse64((unsigned __int64 *)&v5, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v6 = 63 - ((unsigned __int8)v5 ^ 0x3F);
  if ( *(_BYTE *)(a1 + 633) )
  {
    BYTE1(v5) = 1;
    LOBYTE(v5) = 63 - ((unsigned __int8)v5 ^ 0x3F);
    v21 = v4;
    v7 = (unsigned int)sub_2466120(a1, v3, v24, v4, (unsigned __int16)v5, 0);
    v8 = v6;
    v10 = v9;
    BYTE1(v8) = 1;
    v22[0] = "_msld";
    v23 = 259;
    v11 = sub_A82CA0(v24, v21, v7, v8, 0, (__int64)v22);
    sub_246EF60(a1, a2, v11);
  }
  else
  {
    v16 = *(_QWORD *)(a2 + 8);
    v17 = sub_2463540((__int64 *)a1, v16);
    v18 = (__int64)v17;
    if ( v17 )
      v18 = sub_AD6530((__int64)v17, v16);
    v10 = 0;
    sub_246EF60(a1, a2, v18);
  }
  if ( (_BYTE)qword_4FE84C8 )
    sub_2472230(a1, *(_QWORD *)(a2 - 32), a2);
  if ( sub_B46500((unsigned __int8 *)a2) )
  {
    switch ( (*(_WORD *)(a2 + 2) >> 7) & 7 )
    {
      case 0:
        v20 = 0;
        break;
      case 1:
      case 2:
      case 4:
        v20 = 512;
        break;
      case 3:
        BUG();
      case 5:
      case 6:
        v20 = 768;
        break;
      case 7:
        v20 = 896;
        break;
    }
    *(_WORD *)(a2 + 2) = v20 | *(_WORD *)(a2 + 2) & 0xFC7F;
  }
  v12 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v12 + 4) )
  {
    v13 = *(_QWORD *)(v12 + 88);
    if ( *(_BYTE *)(a1 + 633) )
    {
      if ( (unsigned __int8)byte_4FE8EA9 >= v6 )
        v6 = byte_4FE8EA9;
      v23 = 257;
      v14 = v6;
      BYTE1(v14) = 1;
      v15 = sub_A82CA0(v24, v13, v10, v14, 0, (__int64)v22);
      sub_246F1C0(a1, a2, v15);
    }
    else
    {
      v19 = sub_AD6530(*(_QWORD *)(v12 + 88), v13);
      sub_246F1C0(a1, a2, v19);
    }
  }
  nullsub_61();
  v26 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v24[0] != &v25 )
    _libc_free((unsigned __int64)v24[0]);
}
