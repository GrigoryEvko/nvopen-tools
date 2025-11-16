// Function: sub_17DB440
// Address: 0x17db440
//
unsigned __int64 __fastcall sub_17DB440(__int128 a1, char a2)
{
  __int64 v3; // r12
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rax
  _BYTE *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r10
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 result; // rax
  __int64 v26; // rsi
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  _BYTE *v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-E8h]
  __int64 v33; // [rsp+10h] [rbp-E0h]
  __int64 v34; // [rsp+10h] [rbp-E0h]
  __int64 *v35; // [rsp+10h] [rbp-E0h]
  __int64 *v36; // [rsp+18h] [rbp-D8h]
  _QWORD v37[2]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v38[16]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v39; // [rsp+40h] [rbp-B0h]
  _BYTE v40[16]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v41; // [rsp+60h] [rbp-90h]
  __int64 v42[16]; // [rsp+70h] [rbp-80h] BYREF

  v3 = *((_QWORD *)&a1 + 1);
  sub_17CE510((__int64)v42, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v4 = *(_QWORD **)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v4 = (_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v4;
  v36 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(v3 - 8);
  else
    v5 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v5 + 24);
  v6 = sub_17D4DA0(a1);
  v7 = v6;
  if ( a2 )
  {
    v8 = *v6;
    v41 = 257;
    v9 = v8;
    v33 = v8;
    v10 = sub_17CD8D0((_QWORD *)a1, v8);
    v12 = (__int64)v10;
    if ( v10 )
      v12 = sub_15A06D0((__int64 **)v10, v9, v11, (__int64)v10);
    v13 = sub_12AA0C0(v42, 0x21u, v7, v12, (__int64)v40);
    v41 = 257;
    v14 = sub_12AA3B0(v42, 0x26u, v13, v33, (__int64)v40);
  }
  else
  {
    v35 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v3);
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
    {
      v31 = sub_1643360((_QWORD *)v42[3]);
      v7 = (_BYTE *)sub_17CF940((_QWORD *)a1, v42, v7, v31, 1);
    }
    v41 = 257;
    v26 = *(_QWORD *)v7;
    v27 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v7);
    v29 = (__int64)v27;
    if ( v27 )
      v29 = sub_15A06D0((__int64 **)v27, v26, v28, (__int64)v27);
    v30 = (_BYTE *)sub_12AA0C0(v42, 0x21u, v7, v29, (__int64)v40);
    v14 = sub_17CF940((_QWORD *)a1, v42, v30, (__int64)v35, 1);
  }
  v32 = v14;
  v15 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
  v16 = *(__int64 **)(v3 - 24 * v15);
  v17 = *(_QWORD *)(v3 + 24 * (1 - v15));
  v39 = 257;
  v41 = 257;
  v34 = v17;
  v18 = sub_12AA3B0(v42, 0x2Fu, (__int64)v36, *v16, (__int64)v38);
  v19 = *(_QWORD *)(v3 - 24);
  v37[0] = v18;
  v37[1] = v34;
  v20 = sub_1285290(v42, *(_QWORD *)(*(_QWORD *)v19 + 24LL), v19, (int)v37, 2, (__int64)v40, 0);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)v3;
  v41 = 257;
  v21 = v20;
  v22 = sub_17CD8D0((_QWORD *)a1, *((__int64 *)&a1 + 1));
  v23 = sub_12AA3B0(v42, 0x2Fu, v21, (__int64)v22, (__int64)v40);
  v41 = 257;
  v24 = sub_156D390(v42, v23, v32, (__int64)v40);
  sub_17D4920(a1, (__int64 *)v3, v24);
  result = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)a1, v3);
  if ( v42[0] )
    return sub_161E7C0((__int64)v42, v42[0]);
  return result;
}
