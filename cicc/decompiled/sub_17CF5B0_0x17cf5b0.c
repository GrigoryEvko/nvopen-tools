// Function: sub_17CF5B0
// Address: 0x17cf5b0
//
unsigned __int64 __fastcall sub_17CF5B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 **v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 **v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 result; // rax
  __int64 v15; // rax
  __int64 *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rcx
  _QWORD *v19; // rdi
  __int64 v20; // rax
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // [rsp-10h] [rbp-150h]
  char v25[16]; // [rsp+0h] [rbp-140h] BYREF
  __int16 v26; // [rsp+10h] [rbp-130h]
  __int64 v27; // [rsp+20h] [rbp-120h] BYREF
  __int16 v28; // [rsp+30h] [rbp-110h]
  __int64 v29; // [rsp+40h] [rbp-100h] BYREF
  __int16 v30; // [rsp+50h] [rbp-F0h]
  _QWORD v31[4]; // [rsp+60h] [rbp-E0h] BYREF
  char v32[16]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v33; // [rsp+90h] [rbp-B0h]
  _BYTE v34[16]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v35; // [rsp+B0h] [rbp-90h]
  __int64 v36; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v37; // [rsp+C8h] [rbp-78h]
  __int64 *v38; // [rsp+D0h] [rbp-70h]
  _QWORD *v39; // [rsp+D8h] [rbp-68h]

  sub_17CE510((__int64)&v36, a2, 0, 0, 0);
  v33 = 257;
  v26 = 257;
  v4 = sub_16471D0(v39, 0);
  v31[0] = sub_12A95D0(&v36, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v4, (__int64)v25);
  v28 = 257;
  v5 = (__int64 **)sub_1643350(v39);
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a2 + 24 * (1 - v6));
  if ( v5 != *(__int64 ***)v7 )
  {
    if ( *(_BYTE *)(v7 + 16) > 0x10u )
    {
      v35 = 257;
      v15 = sub_15FE0A0((_QWORD *)v7, (__int64)v5, 0, (__int64)v34, 0);
      v7 = v15;
      if ( v37 )
      {
        v16 = v38;
        sub_157E9D0(v37 + 40, v15);
        v17 = *(_QWORD *)(v7 + 24);
        v18 = *v16;
        *(_QWORD *)(v7 + 32) = v16;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v7 + 24) = v18 | v17 & 7;
        *(_QWORD *)(v18 + 8) = v7 + 24;
        *v16 = *v16 & 7 | (v7 + 24);
      }
      sub_164B780(v7, &v27);
      sub_12A86E0(&v36, v7);
      v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    else
    {
      v7 = sub_15A4750(*(__int64 ****)(a2 + 24 * (1 - v6)), v5, 0);
      v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
  }
  v31[1] = v7;
  v8 = *(_QWORD *)(a1 + 8);
  v30 = 257;
  v9 = 3 * (2 - v6);
  v10 = *(__int64 ***)(v8 + 176);
  v11 = *(_QWORD *)(a2 + 8 * v9);
  if ( v10 != *(__int64 ***)v11 )
  {
    if ( *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v19 = *(_QWORD **)(a2 + 8 * v9);
      v35 = 257;
      v20 = sub_15FE0A0(v19, (__int64)v10, 0, (__int64)v34, 0);
      v11 = v20;
      if ( v37 )
      {
        v21 = v38;
        sub_157E9D0(v37 + 40, v20);
        v22 = *(_QWORD *)(v11 + 24);
        v23 = *v21;
        *(_QWORD *)(v11 + 32) = v21;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v11 + 24) = v23 | v22 & 7;
        *(_QWORD *)(v23 + 8) = v11 + 24;
        *v21 = *v21 & 7 | (v11 + 24);
      }
      sub_164B780(v11, &v29);
      sub_12A86E0(&v36, v11);
      v8 = *(_QWORD *)(a1 + 8);
    }
    else
    {
      v12 = sub_15A4750(*(__int64 ****)(a2 + 8 * v9), v10, 0);
      v8 = *(_QWORD *)(a1 + 8);
      v11 = v12;
    }
  }
  v13 = *(_QWORD *)(v8 + 368);
  v31[2] = v11;
  sub_1285290(&v36, *(_QWORD *)(*(_QWORD *)v13 + 24LL), v13, (int)v31, 3, (__int64)v32, 0);
  sub_15F20C0((_QWORD *)a2);
  result = v24;
  if ( v36 )
    return sub_161E7C0((__int64)&v36, v36);
  return result;
}
