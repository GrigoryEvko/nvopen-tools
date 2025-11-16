// Function: sub_AB5480
// Address: 0xab5480
//
__int64 __fastcall sub_AB5480(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v6; // rax
  __int64 *v7; // r15
  unsigned int v8; // ebx
  bool v9; // al
  bool v10; // al
  __int64 *v11; // rax
  __int64 v12; // r15
  unsigned int v13; // ebx
  __int64 *v14; // rbx
  int v15; // eax
  unsigned int v16; // r8d
  unsigned int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // rsi
  __int64 *v24; // r14
  int v25; // eax
  __int64 *v26; // r12
  __int64 *v27; // rbx
  bool v28; // zf
  __int64 *v29; // rax
  __int64 *v30; // r12
  unsigned int v31; // [rsp+0h] [rbp-1A0h]
  __int64 v32; // [rsp+8h] [rbp-198h]
  __int64 v33; // [rsp+8h] [rbp-198h]
  __int64 v34[2]; // [rsp+50h] [rbp-150h] BYREF
  __int64 v35[2]; // [rsp+60h] [rbp-140h] BYREF
  __int64 v36[2]; // [rsp+70h] [rbp-130h] BYREF
  __int64 v37[2]; // [rsp+80h] [rbp-120h] BYREF
  __int64 v38[2]; // [rsp+90h] [rbp-110h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-100h] BYREF
  int v40; // [rsp+A8h] [rbp-F8h]
  __int64 v41[2]; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v42; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v43; // [rsp+D0h] [rbp-D0h] BYREF
  int v44; // [rsp+D8h] [rbp-C8h]
  __int64 v45; // [rsp+E0h] [rbp-C0h] BYREF
  unsigned int v46; // [rsp+E8h] [rbp-B8h]
  __int64 v47[2]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+110h] [rbp-90h] BYREF
  int v50; // [rsp+118h] [rbp-88h]
  __int64 v51; // [rsp+120h] [rbp-80h] BYREF
  __int64 v52; // [rsp+130h] [rbp-70h] BYREF
  int v53; // [rsp+138h] [rbp-68h]
  __int64 v54[2]; // [rsp+140h] [rbp-60h] BYREF
  _BYTE v55[16]; // [rsp+150h] [rbp-50h] BYREF
  _BYTE v56[16]; // [rsp+160h] [rbp-40h] BYREF
  _BYTE v57[48]; // [rsp+170h] [rbp-30h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0((__int64)a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  v6 = sub_9876C0((__int64 *)a2);
  v7 = v6;
  if ( v6 )
  {
    v8 = *((_DWORD *)v6 + 2);
    if ( v8 <= 0x40 )
      v9 = *v6 == 1;
    else
      v9 = v8 - 1 == (unsigned int)sub_C444A0(v6);
    if ( v9 )
    {
      sub_AAF450(a1, (__int64)a3);
      return a1;
    }
    if ( !v8
      || (v8 <= 0x40
        ? (v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) == *v7)
        : (v10 = v8 == (unsigned int)sub_C445E0(v7)),
          v10) )
    {
      sub_9691E0((__int64)&v49, *(_DWORD *)(a2 + 8), 0, 0, 0);
      sub_AADBC0((__int64)&v52, &v49);
      sub_AB51C0(a1, (__int64)&v52, (__int64)a3);
      sub_969240(v54);
      sub_969240(&v52);
      sub_969240(&v49);
      return a1;
    }
  }
  v11 = sub_9876C0(a3);
  v12 = (__int64)v11;
  if ( v11 )
  {
    v13 = *((_DWORD *)v11 + 2);
    if ( v13 <= 0x40 )
    {
      if ( *v11 != 1 )
      {
LABEL_17:
        if ( sub_986760(v12) )
        {
          sub_9691E0((__int64)&v49, *(_DWORD *)(a2 + 8), 0, 0, 0);
          sub_AADBC0((__int64)&v52, &v49);
          sub_AB51C0(a1, (__int64)&v52, a2);
          sub_969240(v54);
          sub_969240(&v52);
          sub_969240(&v49);
          return a1;
        }
        goto LABEL_18;
      }
    }
    else if ( (unsigned int)sub_C444A0(v11) != v13 - 1 )
    {
      goto LABEL_17;
    }
    sub_AAF450(a1, a2);
    return a1;
  }
LABEL_18:
  v14 = &v52;
  sub_AB0A00((__int64)&v52, a2);
  sub_C449B0(v34, &v52, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  sub_969240(&v52);
  sub_AB0910((__int64)&v52, a2);
  sub_C449B0(v35, &v52, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  sub_969240(&v52);
  sub_AB0A00((__int64)&v52, (__int64)a3);
  sub_C449B0(v36, &v52, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  sub_969240(&v52);
  sub_AB0910((__int64)&v52, (__int64)a3);
  sub_C449B0(v37, &v52, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  sub_969240(&v52);
  sub_C472A0(&v49, v35, v37);
  sub_C46A40(&v49, 1);
  v15 = v50;
  v50 = 0;
  v53 = v15;
  v52 = v49;
  sub_C472A0(v47, v34, v36);
  sub_AADC30((__int64)v41, (__int64)v47, &v52);
  sub_969240(v47);
  sub_969240(&v52);
  sub_969240(&v49);
  sub_AB4490((__int64)&v43, (__int64)v41, *(_DWORD *)(a2 + 8));
  if ( sub_AB0100((__int64)&v43) )
    goto LABEL_30;
  v17 = v46;
  v18 = v45;
  v19 = v46 > 0x40 ? *(_QWORD *)(v45 + 8LL * ((v46 - 1) >> 6)) : v45;
  if ( (v19 & (1LL << ((unsigned __int8)v46 - 1))) != 0
    && (v31 = v46, v32 = v45, v22 = sub_986B30(&v45, v45, v46, v19, v16), v18 = v32, v17 = v31, !v22) )
  {
LABEL_30:
    sub_AB14C0((__int64)&v49, a2);
    sub_C44830(&v52, &v49, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    sub_AAD550(v34, &v52);
    sub_969240(&v52);
    sub_969240(&v49);
    sub_AB13A0((__int64)&v49, a2);
    sub_C44830(&v52, &v49, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    sub_AAD550(v35, &v52);
    sub_969240(&v52);
    sub_969240(&v49);
    sub_AB14C0((__int64)&v49, (__int64)a3);
    sub_C44830(&v52, &v49, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    sub_AAD550(v36, &v52);
    sub_969240(&v52);
    sub_969240(&v49);
    v23 = (__int64)a3;
    v24 = v54;
    sub_AB13A0((__int64)&v49, v23);
    sub_C44830(&v52, &v49, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    sub_AAD550(v37, &v52);
    sub_969240(&v52);
    sub_969240(&v49);
    sub_C472A0(&v52, v34, v36);
    sub_C472A0(v54, v34, v37);
    sub_C472A0(v55, v35, v36);
    sub_C472A0(v56, v35, v37);
    do
    {
      if ( (int)sub_C4C880(v14, v24) < 0 )
        v14 = v24;
      v24 += 2;
    }
    while ( v24 != (__int64 *)v57 );
    sub_9865C0((__int64)&v39, (__int64)v14);
    sub_C46A40(&v39, 1);
    v25 = v40;
    v40 = 0;
    v33 = a2;
    v26 = &v52;
    v50 = v25;
    v27 = v54;
    v49 = v39;
    do
    {
      if ( (int)sub_C4C880(v27, v26) < 0 )
        v26 = v27;
      v27 += 2;
    }
    while ( v27 != (__int64 *)v57 );
    sub_9865C0((__int64)v38, (__int64)v26);
    sub_AADC30((__int64)v47, (__int64)v38, &v49);
    sub_969240(v38);
    sub_969240(&v49);
    sub_969240(&v39);
    sub_AB4490((__int64)&v49, (__int64)v47, *(_DWORD *)(v33 + 8));
    v28 = sub_AB01D0((__int64)&v43, (__int64)&v49) == 0;
    v29 = &v43;
    if ( v28 )
      v29 = &v49;
    sub_AAF450(a1, (__int64)v29);
    sub_969240(&v51);
    sub_969240(&v49);
    sub_969240(&v48);
    v30 = (__int64 *)v57;
    sub_969240(v47);
    do
    {
      v30 -= 2;
      sub_969240(v30);
    }
    while ( v30 != &v52 );
  }
  else
  {
    v20 = v44;
    *(_DWORD *)(a1 + 24) = v17;
    v44 = 0;
    *(_DWORD *)(a1 + 8) = v20;
    v21 = v43;
    *(_QWORD *)(a1 + 16) = v18;
    *(_QWORD *)a1 = v21;
    v46 = 0;
  }
  sub_969240(&v45);
  sub_969240(&v43);
  sub_969240(&v42);
  sub_969240(v41);
  sub_969240(v37);
  sub_969240(v36);
  sub_969240(v35);
  sub_969240(v34);
  return a1;
}
