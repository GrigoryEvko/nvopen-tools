// Function: sub_17DA370
// Address: 0x17da370
//
unsigned __int64 __fastcall sub_17DA370(__int128 a1, int a2)
{
  __int64 v2; // r12
  __int64 **v3; // r15
  __int64 v4; // rax
  __int64 *v5; // rdx
  _QWORD *v6; // rax
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 *v14; // rax
  __int64 v15; // rax
  unsigned __int64 result; // rax
  unsigned int v17; // ebx
  __int64 *v18; // rax
  __int64 v19; // [rsp+8h] [rbp-C8h]
  _BYTE *v20; // [rsp+8h] [rbp-C8h]
  _BYTE v21[16]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v22; // [rsp+20h] [rbp-B0h]
  _BYTE v23[16]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v24; // [rsp+40h] [rbp-90h]
  __int64 v25[16]; // [rsp+50h] [rbp-80h] BYREF

  v2 = *((_QWORD *)&a1 + 1);
  if ( *(_BYTE *)(**(_QWORD **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF))
                + 8LL) == 9 )
  {
    v17 = 2 * a2;
    v18 = (__int64 *)sub_1644900(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 168LL), 2 * a2);
    v3 = (__int64 **)sub_16463B0(v18, 0x40 / v17);
  }
  else
  {
    v3 = (__int64 **)**((_QWORD **)&a1 + 1);
  }
  sub_17CE510((__int64)v25, *((__int64 *)&a1 + 1), 0, 0, 0);
  v24 = 257;
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v4 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v4 = *((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v4 + 24);
  v5 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
    v6 = *(_QWORD **)(v2 - 8);
  else
    v6 = (_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v6;
  v19 = (__int64)v5;
  v7 = sub_17D4DA0(a1);
  v8 = sub_156D390(v25, (__int64)v7, v19, (__int64)v23);
  v24 = 257;
  v9 = sub_12AA3B0(v25, 0x2Fu, v8, (__int64)v3, (__int64)v23);
  v24 = 257;
  v22 = 257;
  v20 = (_BYTE *)v9;
  v10 = sub_15A06D0(v3, 47, 257, 257);
  v11 = sub_12AA0C0(v25, 0x21u, v20, v10, (__int64)v21);
  v12 = sub_12AA3B0(v25, 0x26u, v11, (__int64)v3, (__int64)v23);
  v24 = 257;
  v13 = v12;
  v14 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v2);
  v15 = sub_12AA3B0(v25, 0x2Fu, v13, (__int64)v14, (__int64)v23);
  sub_17D4920(a1, (__int64 *)v2, v15);
  result = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)a1, v2);
  if ( v25[0] )
    return sub_161E7C0((__int64)v25, v25[0]);
  return result;
}
