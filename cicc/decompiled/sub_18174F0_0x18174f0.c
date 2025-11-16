// Function: sub_18174F0
// Address: 0x18174f0
//
__int64 __fastcall sub_18174F0(__int64 a1, __int64 a2, _BYTE *a3, unsigned int a4, unsigned int a5, __int64 *a6)
{
  __int64 v9; // r12
  __int64 v11; // rax
  __int64 *v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  bool v16; // cc
  _QWORD *v17; // r15
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // r10
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 *v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rsi
  unsigned __int8 *v30; // rsi
  __int64 v31; // rax
  int v32; // [rsp+4h] [rbp-8Ch]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v36; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v37; // [rsp+30h] [rbp-60h] BYREF
  __int64 v38; // [rsp+38h] [rbp-58h]
  _BYTE v39[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v40; // [rsp+50h] [rbp-40h]

  v9 = a2;
  v11 = sub_1643350(*(_QWORD **)(a1 + 24));
  v12 = (__int64 *)sub_159C470(v11, a4, 0);
  v13 = *(_QWORD **)(a1 + 24);
  v37 = v12;
  v14 = sub_1643350(v13);
  v15 = sub_159C470(v14, a5, 0);
  v16 = a3[16] <= 0x10u;
  v38 = v15;
  if ( v16 )
  {
    v39[4] = 0;
    return sub_15A2E80(a2, (__int64)a3, &v37, 2u, 0, (__int64)v39, 0);
  }
  else
  {
    v40 = 257;
    if ( !a2 )
    {
      v31 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v31 = **(_QWORD **)(v31 + 16);
      v9 = *(_QWORD *)(v31 + 24);
    }
    v19 = sub_1648A60(72, 3u);
    v17 = v19;
    if ( v19 )
    {
      v34 = (__int64)v19;
      v33 = (__int64)(v19 - 9);
      v20 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v20 = **(_QWORD **)(v20 + 16);
      v32 = *(_DWORD *)(v20 + 8) >> 8;
      v21 = (__int64 *)sub_15F9F50(v9, (__int64)&v37, 2);
      v22 = (__int64 *)sub_1646BA0(v21, v32);
      v23 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16
        || (v23 = *v37, *(_BYTE *)(*v37 + 8) == 16)
        || (v23 = *(_QWORD *)v38, *(_BYTE *)(*(_QWORD *)v38 + 8LL) == 16) )
      {
        v22 = sub_16463B0(v22, *(_QWORD *)(v23 + 32));
      }
      sub_15F1EA0((__int64)v17, (__int64)v22, 32, v33, 3, 0);
      v17[7] = v9;
      v17[8] = sub_15F9F50(v9, (__int64)&v37, 2);
      sub_15F9CE0((__int64)v17, (__int64)a3, (__int64 *)&v37, 2, (__int64)v39);
    }
    else
    {
      v34 = 0;
    }
    v24 = *(_QWORD *)(a1 + 8);
    if ( v24 )
    {
      v25 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v24 + 40, (__int64)v17);
      v26 = v17[3];
      v27 = *v25;
      v17[4] = v25;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      v17[3] = v27 | v26 & 7;
      *(_QWORD *)(v27 + 8) = v17 + 3;
      *v25 = *v25 & 7 | (unsigned __int64)(v17 + 3);
    }
    sub_164B780(v34, a6);
    v28 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v36 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v36, v28, 2);
      v29 = v17[6];
      if ( v29 )
        sub_161E7C0((__int64)(v17 + 6), v29);
      v30 = v36;
      v17[6] = v36;
      if ( v30 )
        sub_1623210((__int64)&v36, v30, (__int64)(v17 + 6));
    }
  }
  return (__int64)v17;
}
