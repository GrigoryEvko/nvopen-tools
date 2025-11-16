// Function: sub_17D3460
// Address: 0x17d3460
//
__int64 __fastcall sub_17D3460(__int64 a1, __int64 a2, __int64 a3, char a4, double a5, double a6, double a7)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r9
  bool v14; // cc
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r9
  __int64 *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rcx
  _QWORD *v32; // rax
  __int64 v33; // [rsp+8h] [rbp-B8h]
  __int64 v34; // [rsp+8h] [rbp-B8h]
  __int64 v35; // [rsp+8h] [rbp-B8h]
  char v36[16]; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v37; // [rsp+20h] [rbp-A0h]
  __int64 v38[2]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v39; // [rsp+40h] [rbp-80h]
  __int64 v40[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v41; // [rsp+60h] [rbp-60h]
  _BYTE v42[16]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v43; // [rsp+80h] [rbp-40h]

  if ( a4 )
  {
    v9 = *(_QWORD *)a3;
    v41 = 257;
    v39 = 257;
    v10 = sub_15A0680(v9, 1, 0);
    if ( *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(v10 + 16) > 0x10u )
      v11 = (__int64)sub_17D2EF0((__int64 *)a1, 23, (__int64 *)a3, v10, v40, 0, 0);
    else
      v11 = sub_15A2D50((__int64 *)a3, v10, 0, 0, a5, a6, a7);
    v12 = sub_15A0680(*(_QWORD *)v11, 1, 0);
    if ( *(_BYTE *)(v11 + 16) > 0x10u || *(_BYTE *)(v12 + 16) > 0x10u )
    {
      v43 = 257;
      v32 = (_QWORD *)sub_15FB440(24, (__int64 *)v11, v12, (__int64)v42, 0);
      v13 = (__int64)sub_17CF870((__int64 *)a1, v32, v38);
    }
    else
    {
      v13 = sub_15A2D80((__int64 *)v11, v12, 0, a5, a6, a7);
    }
    v14 = *(_BYTE *)(a3 + 16) <= 0x10u;
    v41 = 257;
    if ( !v14
      || *(_BYTE *)(v13 + 16) > 0x10u
      || (v33 = v13,
          v15 = sub_15A2A30((__int64 *)0x1C, (__int64 *)a3, v13, 0, 0, a5, a6, a7),
          v13 = v33,
          (v16 = v15) == 0) )
    {
      v43 = 257;
      v34 = v13;
      v21 = sub_15FB440(28, (__int64 *)a3, v13, (__int64)v42, 0);
      v22 = *(_QWORD *)(a1 + 8);
      v23 = v34;
      v16 = v21;
      if ( v22 )
      {
        v24 = *(__int64 **)(a1 + 16);
        sub_157E9D0(v22 + 40, v21);
        v25 = *(_QWORD *)(v16 + 24);
        v23 = v34;
        v26 = *v24;
        *(_QWORD *)(v16 + 32) = v24;
        v26 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v16 + 24) = v26 | v25 & 7;
        *(_QWORD *)(v26 + 8) = v16 + 24;
        *v24 = *v24 & 7 | (v16 + 24);
      }
      v35 = v23;
      sub_164B780(v16, v40);
      sub_12A86E0((__int64 *)a1, v16);
      v13 = v35;
    }
    v37 = 257;
    v39 = 257;
    v41 = 257;
    if ( *(_BYTE *)(v13 + 16) > 0x10u )
    {
      v43 = 257;
      v17 = sub_15FB630((__int64 *)v13, (__int64)v42, 0);
      sub_17CCC80(v17, v40, *(_QWORD *)(a1 + 8), *(__int64 **)(a1 + 16));
      sub_12A86E0((__int64 *)a1, v17);
    }
    else
    {
      v17 = sub_15A2B00((__int64 *)v13, a5, a6, a7);
    }
    v18 = sub_1281C00((__int64 *)a1, a2, v17, (__int64)v38);
    return sub_156D390((__int64 *)a1, v18, v16, (__int64)v36);
  }
  else
  {
    v14 = *(_BYTE *)(a3 + 16) <= 0x10u;
    v41 = 257;
    v39 = 257;
    if ( v14 )
    {
      v20 = sub_15A2B00((__int64 *)a3, a5, a6, a7);
    }
    else
    {
      v43 = 257;
      v27 = sub_15FB630((__int64 *)a3, (__int64)v42, 0);
      v28 = *(_QWORD *)(a1 + 8);
      v20 = v27;
      if ( v28 )
      {
        v29 = *(__int64 **)(a1 + 16);
        sub_157E9D0(v28 + 40, v27);
        v30 = *(_QWORD *)(v20 + 24);
        v31 = *v29;
        *(_QWORD *)(v20 + 32) = v29;
        v31 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v20 + 24) = v31 | v30 & 7;
        *(_QWORD *)(v31 + 8) = v20 + 24;
        *v29 = *v29 & 7 | (v20 + 24);
      }
      sub_164B780(v20, v38);
      sub_12A86E0((__int64 *)a1, v20);
    }
    return sub_1281C00((__int64 *)a1, a2, v20, (__int64)v40);
  }
}
