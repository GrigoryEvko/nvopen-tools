// Function: sub_2046710
// Address: 0x2046710
//
__int64 __fastcall sub_2046710(_QWORD *a1, __int64 a2, __int128 *a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned int v6; // edx
  char v7; // al
  __int64 v8; // r14
  __int64 *v9; // r15
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // rax
  bool v13; // zf
  _QWORD *v14; // rbx
  int v15; // r13d
  unsigned int v16; // edx
  unsigned int v18; // [rsp+0h] [rbp-80h] BYREF
  __int64 v19; // [rsp+8h] [rbp-78h]
  __int128 v20; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  _QWORD v22[3]; // [rsp+30h] [rbp-50h] BYREF

  v4 = a1[2];
  v5 = sub_1E0A0C0(a1[4]);
  v6 = 8 * sub_15A9520(v5, 0);
  if ( v6 == 32 )
  {
    v7 = 5;
  }
  else if ( v6 > 0x20 )
  {
    v7 = 6;
    if ( v6 != 64 )
    {
      v7 = 0;
      if ( v6 == 128 )
        v7 = 7;
    }
  }
  else
  {
    v7 = 3;
    if ( v6 != 8 )
      v7 = 4 * (v6 == 16);
  }
  v8 = a1[4];
  LOBYTE(v18) = v7;
  v19 = 0;
  v9 = (__int64 *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 536LL))(
                    v4,
                    *(_QWORD *)(*(_QWORD *)v8 + 40LL));
  v11 = sub_1D2CC80(a1, 22, a2, v18, 0, v10, *a3);
  if ( v9 )
  {
    v12 = *v9;
    v20 = (unsigned __int64)v9;
    v13 = *(_BYTE *)(v12 + 8) == 16;
    LOBYTE(v21) = 0;
    if ( v13 )
      v12 = **(_QWORD **)(v12 + 16);
    HIDWORD(v21) = *(_DWORD *)(v12 + 8) >> 8;
    memset(v22, 0, sizeof(v22));
    v14 = (_QWORD *)sub_1E0A240(v8, 1);
    v15 = sub_1D172F0((__int64)a1, v18, v19);
    if ( (_BYTE)v18 )
      v16 = sub_2045180(v18);
    else
      v16 = sub_1F58D40((__int64)&v18);
    *v14 = sub_1E0B8E0(v8, 0x31u, v16 >> 3, v15, (int)v22, 0, v20, v21, 1u, 0, 0);
    *(_QWORD *)(v11 + 88) = v14;
    *(_QWORD *)(v11 + 96) = v14 + 1;
  }
  return v11;
}
