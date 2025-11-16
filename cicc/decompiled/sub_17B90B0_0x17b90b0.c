// Function: sub_17B90B0
// Address: 0x17b90b0
//
__int64 __fastcall sub_17B90B0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // rax
  __int64 *v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // cc
  _QWORD *v14; // r15
  __int64 v16; // rax
  __int64 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // r10
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 *v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  int v30; // [rsp+4h] [rbp-8Ch]
  __int64 v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v34; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v35; // [rsp+30h] [rbp-60h] BYREF
  __int64 v36; // [rsp+38h] [rbp-58h]
  _BYTE v37[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v38; // [rsp+50h] [rbp-40h]

  v8 = sub_1643360(*(_QWORD **)(a1 + 24));
  v9 = (__int64 *)sub_159C470(v8, a3, 0);
  v10 = *(_QWORD **)(a1 + 24);
  v35 = v9;
  v11 = sub_1643360(v10);
  v12 = sub_159C470(v11, a4, 0);
  v13 = a2[16] <= 0x10u;
  v36 = v12;
  if ( v13 )
  {
    v37[4] = 0;
    return sub_15A2E80(0, (__int64)a2, &v35, 2u, 1u, (__int64)v37, 0);
  }
  else
  {
    v38 = 257;
    v16 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v16 = **(_QWORD **)(v16 + 16);
    v17 = *(_QWORD *)(v16 + 24);
    v18 = sub_1648A60(72, 3u);
    v14 = v18;
    if ( v18 )
    {
      v32 = (__int64)v18;
      v31 = (__int64)(v18 - 9);
      v19 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v19 = **(_QWORD **)(v19 + 16);
      v30 = *(_DWORD *)(v19 + 8) >> 8;
      v20 = (__int64 *)sub_15F9F50(v17, (__int64)&v35, 2);
      v21 = (__int64 *)sub_1646BA0(v20, v30);
      v22 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16
        || (v22 = *v35, *(_BYTE *)(*v35 + 8) == 16)
        || (v22 = *(_QWORD *)v36, *(_BYTE *)(*(_QWORD *)v36 + 8LL) == 16) )
      {
        v21 = sub_16463B0(v21, *(_QWORD *)(v22 + 32));
      }
      sub_15F1EA0((__int64)v14, (__int64)v21, 32, v31, 3, 0);
      v14[7] = v17;
      v14[8] = sub_15F9F50(v17, (__int64)&v35, 2);
      sub_15F9CE0((__int64)v14, (__int64)a2, (__int64 *)&v35, 2, (__int64)v37);
    }
    else
    {
      v32 = 0;
    }
    sub_15FA2E0((__int64)v14, 1);
    v23 = *(_QWORD *)(a1 + 8);
    if ( v23 )
    {
      v24 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v23 + 40, (__int64)v14);
      v25 = v14[3];
      v26 = *v24;
      v14[4] = v24;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      v14[3] = v26 | v25 & 7;
      *(_QWORD *)(v26 + 8) = v14 + 3;
      *v24 = *v24 & 7 | (unsigned __int64)(v14 + 3);
    }
    sub_164B780(v32, a5);
    v27 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v34 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v34, v27, 2);
      v28 = v14[6];
      if ( v28 )
        sub_161E7C0((__int64)(v14 + 6), v28);
      v29 = v34;
      v14[6] = v34;
      if ( v29 )
        sub_1623210((__int64)&v34, v29, (__int64)(v14 + 6));
    }
  }
  return (__int64)v14;
}
