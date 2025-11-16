// Function: sub_17DB110
// Address: 0x17db110
//
unsigned __int64 __fastcall sub_17DB110(__int128 a1, double a2, double a3, double a4)
{
  __int64 v4; // rbx
  __int64 **v5; // r14
  __int64 v6; // rax
  __int64 *v7; // rdx
  _QWORD *v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 *v20; // rax
  __int64 v21; // rax
  unsigned __int64 result; // rax
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  _BYTE *v26; // [rsp+8h] [rbp-D8h]
  __int64 v27; // [rsp+10h] [rbp-D0h]
  unsigned int v28; // [rsp+18h] [rbp-C8h]
  __int64 *v29; // [rsp+18h] [rbp-C8h]
  __int64 v30[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v31; // [rsp+30h] [rbp-B0h]
  _BYTE v32[16]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v33; // [rsp+50h] [rbp-90h]
  __int64 v34; // [rsp+60h] [rbp-80h] BYREF
  __int64 v35; // [rsp+68h] [rbp-78h]
  __int64 *v36; // [rsp+70h] [rbp-70h]

  v4 = *((_QWORD *)&a1 + 1);
  if ( *(_BYTE *)(**(_QWORD **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF))
                + 8LL) == 9 )
    v5 = (__int64 **)sub_1644900(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 168LL), 0x40u);
  else
    v5 = (__int64 **)**((_QWORD **)&a1 + 1);
  v28 = sub_16431D0((__int64)v5) - 16;
  sub_17CE510((__int64)&v34, *((__int64 *)&a1 + 1), 0, 0, 0);
  v33 = 257;
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v6 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v6 = *((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v6 + 24);
  v7 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
    v8 = *(_QWORD **)(v4 - 8);
  else
    v8 = (_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v8;
  v27 = (__int64)v7;
  v9 = sub_17D4DA0(a1);
  v10 = sub_156D390(&v34, (__int64)v9, v27, (__int64)v32);
  v33 = 257;
  v11 = sub_12AA3B0(&v34, 0x2Fu, v10, (__int64)v5, (__int64)v32);
  v33 = 257;
  v31 = 257;
  v26 = (_BYTE *)v11;
  v14 = sub_15A06D0(v5, 47, v12, v13);
  v15 = sub_12AA0C0(&v34, 0x21u, v26, v14, (__int64)v30);
  v16 = (__int64 *)sub_12AA3B0(&v34, 0x26u, v15, (__int64)v5, (__int64)v32);
  v31 = 257;
  v17 = v16;
  v18 = sub_15A0680(*v16, v28, 0);
  if ( *((_BYTE *)v17 + 16) > 0x10u || *(_BYTE *)(v18 + 16) > 0x10u )
  {
    v33 = 257;
    v23 = sub_15FB440(24, v17, v18, (__int64)v32, 0);
    v19 = v23;
    if ( v35 )
    {
      v29 = v36;
      sub_157E9D0(v35 + 40, v23);
      v24 = *v29;
      v25 = *(_QWORD *)(v19 + 24) & 7LL;
      *(_QWORD *)(v19 + 32) = v29;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v19 + 24) = v24 | v25;
      *(_QWORD *)(v24 + 8) = v19 + 24;
      *v29 = *v29 & 7 | (v19 + 24);
    }
    sub_164B780(v19, v30);
    sub_12A86E0(&v34, v19);
  }
  else
  {
    v19 = sub_15A2D80(v17, v18, 0, a2, a3, a4);
  }
  *((_QWORD *)&a1 + 1) = *(_QWORD *)v4;
  v33 = 257;
  v20 = sub_17CD8D0((_QWORD *)a1, *((__int64 *)&a1 + 1));
  v21 = sub_12AA3B0(&v34, 0x2Fu, v19, (__int64)v20, (__int64)v32);
  sub_17D4920(a1, (__int64 *)v4, v21);
  result = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)a1, v4);
  if ( v34 )
    return sub_161E7C0((__int64)&v34, v34);
  return result;
}
