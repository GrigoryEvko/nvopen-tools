// Function: sub_17DAA00
// Address: 0x17daa00
//
unsigned __int64 __fastcall sub_17DAA00(__int128 a1, double a2, double a3, double a4)
{
  __int128 v4; // kr00_16
  _QWORD *v5; // rax
  __int64 *v6; // r15
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // edi
  __int64 v16; // rdx
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  unsigned __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  char v25; // al
  int v26; // r15d
  __int64 *v27; // r15
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-E0h]
  __int64 v31; // [rsp+8h] [rbp-D8h]
  __int64 v32; // [rsp+8h] [rbp-D8h]
  __int64 v33; // [rsp+8h] [rbp-D8h]
  __int64 v34; // [rsp+10h] [rbp-D0h]
  __int64 v35; // [rsp+18h] [rbp-C8h]
  __int64 v36; // [rsp+18h] [rbp-C8h]
  __int64 v37[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v38; // [rsp+30h] [rbp-B0h]
  _BYTE v39[16]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v40; // [rsp+50h] [rbp-90h]
  __int64 v41; // [rsp+60h] [rbp-80h] BYREF
  __int64 v42; // [rsp+68h] [rbp-78h]
  __int64 *v43; // [rsp+70h] [rbp-70h]
  __int64 v44; // [rsp+80h] [rbp-60h]
  int v45; // [rsp+88h] [rbp-58h]

  v4 = a1;
  sub_17CE510((__int64)&v41, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v5 = *(_QWORD **)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v5 = (_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v5;
  v6 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(*((_QWORD *)&v4 + 1) + 23LL) & 0x40) != 0 )
    v7 = *(_QWORD *)(*((_QWORD *)&v4 + 1) - 8LL);
  else
    v7 = *((_QWORD *)&v4 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&v4 + 1) + 20LL) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v7 + 24);
  v8 = sub_17D4DA0(a1);
  v40 = 257;
  v9 = *v8;
  v38 = 257;
  *((_QWORD *)&a1 + 1) = *v8;
  v35 = v9;
  v10 = sub_17CD8D0((_QWORD *)a1, *v8);
  v12 = (__int64)v10;
  if ( v10 )
    v12 = sub_15A06D0((__int64 **)v10, *((__int64 *)&a1 + 1), v11, (__int64)v10);
  v13 = sub_12AA0C0(&v41, 0x21u, v8, v12, (__int64)v37);
  v14 = sub_12AA3B0(&v41, 0x26u, v13, v35, (__int64)v39);
  v15 = *(unsigned __int8 *)(*((_QWORD *)&v4 + 1) + 16LL);
  v16 = *(_QWORD *)(*((_QWORD *)&v4 + 1) - 24LL);
  v38 = 257;
  v17 = (__int64 *)(unsigned int)(v15 - 24);
  v36 = v14;
  if ( *((_BYTE *)v6 + 16) > 0x10u
    || *(_BYTE *)(v16 + 16) > 0x10u
    || (v30 = v16, v18 = sub_15A2A30(v17, v6, v16, 0, 0, a2, a3, a4), v16 = v30, (v19 = v18) == 0) )
  {
    v40 = 257;
    v22 = sub_15FB440((int)v17, v6, v16, (__int64)v39, 0);
    v23 = *(_QWORD *)v22;
    v24 = v22;
    v25 = *(_BYTE *)(*(_QWORD *)v22 + 8LL);
    if ( v25 == 16 )
      v25 = *(_BYTE *)(**(_QWORD **)(v23 + 16) + 8LL);
    if ( (unsigned __int8)(v25 - 1) <= 5u || *(_BYTE *)(v24 + 16) == 76 )
    {
      v26 = v45;
      if ( v44 )
      {
        v31 = v24;
        sub_1625C10(v24, 3, v44);
        v24 = v31;
      }
      v32 = v24;
      sub_15F2440(v24, v26);
      v24 = v32;
    }
    if ( v42 )
    {
      v27 = v43;
      v33 = v24;
      sub_157E9D0(v42 + 40, v24);
      v24 = v33;
      v28 = *v27;
      v29 = *(_QWORD *)(v33 + 24);
      *(_QWORD *)(v33 + 32) = v27;
      v28 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v33 + 24) = v28 | v29 & 7;
      *(_QWORD *)(v28 + 8) = v33 + 24;
      *v27 = *v27 & 7 | (v33 + 24);
    }
    v34 = v24;
    sub_164B780(v24, v37);
    sub_12A86E0(&v41, v34);
    v19 = v34;
  }
  v40 = 257;
  v20 = sub_156D390(&v41, v19, v36, (__int64)v39);
  sub_17D4920(v4, *((__int64 **)&v4 + 1), v20);
  result = *(_QWORD *)(v4 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)v4, *((__int64 *)&v4 + 1));
  if ( v41 )
    return sub_161E7C0((__int64)&v41, v41);
  return result;
}
