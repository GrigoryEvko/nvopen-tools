// Function: sub_156B790
// Address: 0x156b790
//
__int64 __fastcall sub_156B790(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // rdi
  unsigned __int64 *v25; // r13
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // [rsp+0h] [rbp-70h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  _QWORD v35[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v36; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
    return sub_15A2DC0(a2, a3, a4, 0);
  v36 = 257;
  v10 = sub_1648A60(56, 3);
  v11 = (_QWORD *)v10;
  if ( v10 )
  {
    v30 = a4;
    v31 = v10 - 72;
    v34 = v10;
    sub_15F1EA0(v10, *(_QWORD *)a3, 55, v10 - 72, 3, 0);
    if ( *(v11 - 9) )
    {
      v12 = *(v11 - 8);
      v13 = *(v11 - 7) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v13 = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
    }
    *(v11 - 9) = a2;
    v14 = *(_QWORD *)(a2 + 8);
    *(v11 - 8) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = (unsigned __int64)(v11 - 8) | *(_QWORD *)(v14 + 16) & 3LL;
    *(v11 - 7) = (a2 + 8) | *(v11 - 7) & 3LL;
    *(_QWORD *)(a2 + 8) = v31;
    if ( *(v11 - 6) )
    {
      v15 = *(v11 - 5);
      v16 = *(v11 - 4) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
    }
    *(v11 - 6) = a3;
    v17 = *(_QWORD *)(a3 + 8);
    *(v11 - 5) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = (unsigned __int64)(v11 - 5) | *(_QWORD *)(v17 + 16) & 3LL;
    *(v11 - 4) = (a3 + 8) | *(v11 - 4) & 3LL;
    *(_QWORD *)(a3 + 8) = v11 - 6;
    if ( *(v11 - 3) )
    {
      v18 = *(v11 - 2);
      v19 = *(v11 - 1) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v19 = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
    }
    *(v11 - 3) = v30;
    if ( v30 )
    {
      v20 = *(_QWORD *)(v30 + 8);
      *(v11 - 2) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = (unsigned __int64)(v11 - 2) | *(_QWORD *)(v20 + 16) & 3LL;
      *(v11 - 1) = (v30 + 8) | *(v11 - 1) & 3LL;
      *(_QWORD *)(v30 + 8) = v11 - 3;
    }
    sub_164B780(v11, v35);
  }
  else
  {
    v34 = 0;
  }
  if ( a6 && (*(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0) )
  {
    v21 = sub_1625790(a6, 2);
    v22 = v21;
    if ( *(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0 )
    {
      v23 = sub_1625790(a6, 15);
      if ( v22 )
        sub_1625C10(v34, 2, v22);
      if ( v23 )
        sub_1625C10(v34, 15, v23);
    }
    else if ( v21 )
    {
      sub_1625C10(v34, 2, v21);
    }
  }
  v24 = a1[1];
  if ( v24 )
  {
    v25 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v24 + 40, v11);
    v26 = v11[3];
    v27 = *v25;
    v11[4] = v25;
    v27 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v27 | v26 & 7;
    *(_QWORD *)(v27 + 8) = v11 + 3;
    *v25 = *v25 & 7 | (unsigned __int64)(v11 + 3);
  }
  sub_164B780(v34, a5);
  v28 = *a1;
  if ( *a1 )
  {
    v35[0] = *a1;
    sub_1623A60(v35, v28, 2);
    if ( v11[6] )
      sub_161E7C0(v11 + 6);
    v29 = v35[0];
    v11[6] = v35[0];
    if ( v29 )
      sub_1623210(v35, v29, v11 + 6);
  }
  return (__int64)v11;
}
