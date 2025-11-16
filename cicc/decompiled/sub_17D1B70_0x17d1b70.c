// Function: sub_17D1B70
// Address: 0x17d1b70
//
unsigned __int64 __fastcall sub_17D1B70(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // ecx
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  char v30; // [rsp+17h] [rbp-B9h]
  _QWORD v32[2]; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int8 *v33[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v34; // [rsp+40h] [rbp-90h]
  __int64 v35[3]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v36; // [rsp+68h] [rbp-68h]

  v7 = a2;
  v30 = a5;
  sub_17CE510((__int64)v35, a2, 0, 0, 0);
  v10 = *(_QWORD *)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
  {
    a2 = *(_DWORD *)(v10 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(v10 + 24));
    v9 = sub_1644900(*(_QWORD **)(a1[1] + 168LL), a2);
    if ( v10 != v9 )
    {
      v34 = 257;
      a2 = 47;
      a3 = sub_12AA3B0(v35, 0x2Fu, a3, v9, (__int64)v33);
      if ( !a3 )
      {
        v5 = sub_15F2050(v7);
        sub_1632FA0(v5);
        BUG();
      }
    }
  }
  if ( *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v12 = sub_15F2050(v7);
    v13 = sub_1632FA0(v12);
    v14 = sub_127FA20(v13, *(_QWORD *)a3);
    v15 = 0;
    if ( v14 > 8 )
    {
      v15 = ((v14 + 7) >> 3) - 1;
      if ( (v14 + 7) >> 3 != 1 )
      {
        _BitScanReverse(&v16, v15);
        v15 = 32 - (v16 ^ 0x1F);
        v30 = a5 & (v15 <= 3);
      }
    }
    if ( !v30 )
    {
      v33[0] = "_mscmp";
      v34 = 259;
      v17 = sub_17CDAE0(a1, *(_QWORD *)a3);
      v18 = sub_12AA0C0(v35, 0x21u, (_BYTE *)a3, v17, (__int64)v33);
      v19 = sub_1AA92B0(v18, v7, *(_BYTE *)(a1[1] + 160LL) ^ 1u, *(_QWORD *)(a1[1] + 416LL), 0, 0);
      v20 = *(unsigned __int8 **)(v19 + 48);
      v35[1] = *(_QWORD *)(v19 + 40);
      v35[2] = v19 + 24;
      v33[0] = v20;
      if ( v20 )
        sub_1623A60((__int64)v33, (__int64)v20, 2);
      sub_17CD270(v35);
      v35[0] = (__int64)v33[0];
      if ( v33[0] )
      {
        sub_1623210((__int64)v33, v33[0], (__int64)v35);
        v33[0] = 0;
      }
      sub_17CD270((__int64 *)v33);
      if ( !a4 )
      {
        v27 = sub_1643350(v36);
        a4 = sub_159C470(v27, 0, 0);
      }
      v21 = a1[1];
      if ( *(_DWORD *)(v21 + 156) )
      {
        sub_12A8F50(v35, a4, *(_QWORD *)(v21 + 240), 0);
        v21 = a1[1];
      }
      goto LABEL_19;
    }
    v23 = *(_QWORD *)(a1[1] + 8LL * v15 + 264);
    v34 = 257;
    v24 = sub_1644C60(v36, 8 << v15);
    v32[0] = sub_12AA3B0(v35, 0x25u, a3, v24, (__int64)v33);
    v25 = a1[1];
    v34 = 257;
    if ( !*(_DWORD *)(v25 + 156) || !a4 )
    {
      v26 = sub_1643350(v36);
      a4 = sub_159C470(v26, 0, 0);
    }
    v32[1] = a4;
    sub_1285290(v35, *(_QWORD *)(*(_QWORD *)v23 + 24LL), v23, (int)v32, 2, (__int64)v33, 0);
  }
  else if ( byte_4FA4360 && !sub_1595F50(a3, a2, v8, v9) )
  {
    if ( !a4 )
    {
      v28 = sub_1643350(v36);
      a4 = sub_159C470(v28, 0, 0);
    }
    v21 = a1[1];
    if ( *(_DWORD *)(v21 + 156) )
    {
      sub_12A8F50(v35, a4, *(_QWORD *)(v21 + 240), 0);
      v21 = a1[1];
    }
LABEL_19:
    v34 = 257;
    sub_1285290(v35, *(_QWORD *)(**(_QWORD **)(v21 + 256) + 24LL), *(_QWORD *)(v21 + 256), 0, 0, (__int64)v33, 0);
    v22 = a1[1];
    v34 = 257;
    sub_1285290(v35, *(_QWORD *)(**(_QWORD **)(v22 + 432) + 24LL), *(_QWORD *)(v22 + 432), 0, 0, (__int64)v33, 0);
  }
  return sub_17CD270(v35);
}
