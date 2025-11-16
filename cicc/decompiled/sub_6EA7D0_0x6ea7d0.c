// Function: sub_6EA7D0
// Address: 0x6ea7d0
//
__int64 __fastcall sub_6EA7D0(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        _QWORD *a9)
{
  __int64 v9; // r15
  int v10; // r14d
  char v13; // al
  __int64 v14; // r14
  __int64 v15; // rax
  char i; // dl
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  char v23; // di
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  _QWORD *v29; // r8
  int v30; // [rsp+14h] [rbp-4Ch]
  __int64 v32; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v33[7]; // [rsp+28h] [rbp-38h] BYREF

  v9 = a1;
  v10 = a6;
  v13 = *(_BYTE *)(a1 + 80);
  v30 = a5;
  if ( v13 == 16 )
  {
    v9 = **(_QWORD **)(a1 + 88);
    v13 = *(_BYTE *)(v9 + 80);
  }
  if ( v13 == 24 )
    v9 = *(_QWORD *)(v9 + 88);
  v32 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( a7 && v10 )
    goto LABEL_7;
  if ( !dword_4D04964 )
  {
    if ( dword_4F077BC )
    {
      sub_6E5C80(7, 0x1F8u, a3);
      goto LABEL_7;
    }
    if ( qword_4D0495C )
      goto LABEL_9;
    v23 = 7;
    goto LABEL_24;
  }
  v23 = unk_4F07471;
  if ( unk_4F07471 != 3 )
LABEL_24:
    sub_6E5C80(v23, 0x1F8u, a3);
LABEL_7:
  if ( !qword_4D0495C && v30 && (unsigned int)sub_6E6010() )
  {
    LODWORD(v33[0]) = 0;
    v29 = 0;
    if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
      v29 = v33;
    sub_8843A0(a1, a2, a3, *(_QWORD *)(a2 + 64), v29);
    if ( LODWORD(v33[0]) )
      sub_6E50A0();
  }
LABEL_9:
  v14 = *(_QWORD *)(v9 + 88);
  if ( *(_BYTE *)(v9 + 80) != 8 )
  {
    if ( (*(_BYTE *)(v14 + 207) & 0x30) == 0x10 )
      sub_8B1A30(*(_QWORD *)(v9 + 88), a3);
    v15 = *(_QWORD *)(v14 + 152);
    for ( i = *(_BYTE *)(v15 + 140); i == 12; i = *(_BYTE *)(v15 + 140) )
      v15 = *(_QWORD *)(v15 + 160);
    if ( i == 7 && (*(_BYTE *)(*(_QWORD *)(v15 + 168) + 20LL) & 2) != 0 )
      sub_6E5470(*(_QWORD *)(v15 + 104), a3);
    v17 = v32;
    sub_73F170(v14, v32);
    if ( (*(_BYTE *)(v14 + 192) & 2) == 0 )
      sub_6E1D20((__int64 *)v14, v32, v18, v19, v20);
LABEL_17:
    if ( dword_4F04C44 != -1
      || (v21 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v21 + 6) & 6) != 0)
      || *(_BYTE *)(v21 + 4) == 12 )
    {
      v24 = *(_QWORD *)(v32 + 128);
      if ( (unsigned int)sub_8DBE70(v24) )
      {
        v33[0] = sub_724DC0(v24, v17, v25, v26, v27, v28);
        sub_72A510(v32, v33[0]);
        sub_70FDD0(v33[0], v32, *(_QWORD *)(v32 + 128), 0);
        sub_724E30(v33);
      }
    }
    sub_6E6A50(v32, (__int64)a9);
    goto LABEL_21;
  }
  if ( !(unsigned int)sub_8D32E0(*(_QWORD *)(v14 + 120)) )
  {
    if ( (*(_BYTE *)(v14 + 144) & 4) != 0 && (unsigned int)sub_6E5430() )
      sub_6851C0(0x8Bu, a3);
    v17 = v32;
    sub_73F1E0(v14, v32);
    goto LABEL_17;
  }
  if ( (unsigned int)sub_6E5430() )
    sub_685360(0x22Eu, a3, *(_QWORD *)(v14 + 120));
  sub_6E6260(a9);
LABEL_21:
  *(_QWORD *)((char *)a9 + 68) = *(_QWORD *)a3;
  *(_QWORD *)((char *)a9 + 76) = *a4;
  return sub_724E30(&v32);
}
