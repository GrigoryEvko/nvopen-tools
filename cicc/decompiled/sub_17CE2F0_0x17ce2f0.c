// Function: sub_17CE2F0
// Address: 0x17ce2f0
//
__int64 __fastcall sub_17CE2F0(__int64 a1, __int64 a2, _BYTE *a3, unsigned int a4, __int64 *a5)
{
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 *v10; // rax
  bool v11; // cc
  _QWORD *v12; // rbx
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // r10
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int64 *v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rax
  int v24; // [rsp+4h] [rbp-7Ch]
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 *v28; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v29[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v7 = a2;
  v9 = sub_1643350(*(_QWORD **)(a1 + 24));
  v10 = (__int64 *)sub_159C470(v9, a4, 0);
  v11 = a3[16] <= 0x10u;
  v28 = v10;
  if ( v11 )
  {
    v29[4] = 0;
    return sub_15A2E80(a2, (__int64)a3, &v28, 1u, 0, (__int64)v29, 0);
  }
  else
  {
    v30 = 257;
    if ( !a2 )
    {
      v23 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v23 = **(_QWORD **)(v23 + 16);
      v7 = *(_QWORD *)(v23 + 24);
    }
    v14 = sub_1648A60(72, 2u);
    v12 = v14;
    if ( v14 )
    {
      v26 = (__int64)v14;
      v25 = (__int64)(v14 - 6);
      v15 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v15 = **(_QWORD **)(v15 + 16);
      v24 = *(_DWORD *)(v15 + 8) >> 8;
      v16 = (__int64 *)sub_15F9F50(v7, (__int64)&v28, 1);
      v17 = (__int64 *)sub_1646BA0(v16, v24);
      v18 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 || (v18 = *v28, *(_BYTE *)(*v28 + 8) == 16) )
        v17 = sub_16463B0(v17, *(_QWORD *)(v18 + 32));
      sub_15F1EA0((__int64)v12, (__int64)v17, 32, v25, 2, 0);
      v12[7] = v7;
      v12[8] = sub_15F9F50(v7, (__int64)&v28, 1);
      sub_15F9CE0((__int64)v12, (__int64)a3, (__int64 *)&v28, 1, (__int64)v29);
    }
    else
    {
      v26 = 0;
    }
    v19 = *(_QWORD *)(a1 + 8);
    if ( v19 )
    {
      v20 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v19 + 40, (__int64)v12);
      v21 = v12[3];
      v22 = *v20;
      v12[4] = v20;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      v12[3] = v22 | v21 & 7;
      *(_QWORD *)(v22 + 8) = v12 + 3;
      *v20 = *v20 & 7 | (unsigned __int64)(v12 + 3);
    }
    sub_164B780(v26, a5);
    sub_12A86E0((__int64 *)a1, (__int64)v12);
  }
  return (__int64)v12;
}
