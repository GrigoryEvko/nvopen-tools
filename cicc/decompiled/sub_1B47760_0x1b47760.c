// Function: sub_1B47760
// Address: 0x1b47760
//
_QWORD *__fastcall sub_1B47760(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  _QWORD *v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rdi
  unsigned __int64 *v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  _QWORD *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // rdi
  unsigned __int64 *v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  _QWORD *v37; // [rsp+0h] [rbp-70h]
  _QWORD *v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 v42[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v43; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v43 = 257;
    v17 = sub_1648A60(56, 3u);
    v10 = v17;
    if ( v17 )
    {
      v37 = v17 - 9;
      v39 = (__int64)v17;
      sub_15F1EA0((__int64)v17, *(_QWORD *)a3, 55, (__int64)(v17 - 9), 3, 0);
      if ( *(v10 - 9) )
      {
        v18 = *(v10 - 8);
        v19 = *(v10 - 7) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *(v10 - 9) = a2;
      v20 = *(_QWORD *)(a2 + 8);
      *(v10 - 8) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = (unsigned __int64)(v10 - 8) | *(_QWORD *)(v20 + 16) & 3LL;
      *(v10 - 7) = (a2 + 8) | *(v10 - 7) & 3LL;
      *(_QWORD *)(a2 + 8) = v37;
      if ( *(v10 - 6) )
      {
        v21 = *(v10 - 5);
        v22 = *(v10 - 4) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v22 = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
      }
      *(v10 - 6) = a3;
      v23 = *(_QWORD *)(a3 + 8);
      *(v10 - 5) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = (unsigned __int64)(v10 - 5) | *(_QWORD *)(v23 + 16) & 3LL;
      *(v10 - 4) = (a3 + 8) | *(v10 - 4) & 3LL;
      *(_QWORD *)(a3 + 8) = v10 - 6;
      if ( *(v10 - 3) )
      {
        v24 = *(v10 - 2);
        v25 = *(v10 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v25 = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = *(_QWORD *)(v24 + 16) & 3LL | v25;
      }
      *(v10 - 3) = a4;
      if ( a4 )
      {
        v26 = *(_QWORD *)(a4 + 8);
        *(v10 - 2) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = (unsigned __int64)(v10 - 2) | *(_QWORD *)(v26 + 16) & 3LL;
        *(v10 - 1) = (a4 + 8) | *(v10 - 1) & 3LL;
        *(_QWORD *)(a4 + 8) = v10 - 3;
      }
      sub_164B780((__int64)v10, v42);
    }
    else
    {
      v39 = 0;
    }
    if ( a6 && (*(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0) )
    {
      v27 = sub_1625790(a6, 2);
      v28 = v27;
      if ( *(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0 )
      {
        v29 = sub_1625790(a6, 15);
        if ( v28 )
          sub_1625C10(v39, 2, v28);
        if ( v29 )
          sub_1625C10(v39, 15, v29);
      }
      else if ( v27 )
      {
        sub_1625C10(v39, 2, v27);
      }
    }
    v30 = a1[1];
    if ( v30 )
    {
      v31 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v30 + 40, (__int64)v10);
      v32 = v10[3];
      v33 = *v31;
      v10[4] = v31;
      v33 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v33 | v32 & 7;
      *(_QWORD *)(v33 + 8) = v10 + 3;
      *v31 = *v31 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780(v39, a5);
    v34 = *a1;
    if ( *a1 )
    {
      v42[0] = *a1;
      sub_1623A60((__int64)v42, v34, 2);
      v35 = v10[6];
      if ( v35 )
        sub_161E7C0((__int64)(v10 + 6), v35);
      v36 = (unsigned __int8 *)v42[0];
      v10[6] = v42[0];
      if ( v36 )
        sub_1623210((__int64)v42, v36, (__int64)(v10 + 6));
    }
  }
  else
  {
    v43 = 257;
    v9 = sub_1648A60(56, 3u);
    v10 = v9;
    if ( v9 )
    {
      v11 = (__int64)v9;
      v38 = v9 - 9;
      sub_15F1EA0((__int64)v9, *(_QWORD *)a3, 55, (__int64)(v9 - 9), 3, 0);
      sub_1593B40(v38, a2);
      sub_1593B40(v10 - 6, a3);
      sub_1593B40(v10 - 3, a4);
      sub_164B780((__int64)v10, v42);
    }
    else
    {
      v11 = 0;
    }
    v12 = a1[1];
    if ( v12 )
    {
      v13 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v12 + 40, (__int64)v10);
      v14 = v10[3];
      v15 = *v13;
      v10[4] = v13;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v15 | v14 & 7;
      *(_QWORD *)(v15 + 8) = v10 + 3;
      *v13 = *v13 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780(v11, a5);
    sub_12A86E0(a1, (__int64)v10);
  }
  return v10;
}
