// Function: sub_1707C10
// Address: 0x1707c10
//
_QWORD *__fastcall sub_1707C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  _QWORD *v9; // r12
  __int64 v10; // rax
  _QWORD *v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r14
  __int64 v25; // rdi
  unsigned __int64 *v26; // r13
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rdx
  bool v30; // zf
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // [rsp+0h] [rbp-70h]
  _QWORD *v35; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+18h] [rbp-58h]
  __int64 v39[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v40; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v40 = 257;
    v12 = sub_1648A60(56, 3u);
    v9 = v12;
    if ( v12 )
    {
      v34 = a4;
      v35 = v12 - 9;
      v38 = (__int64)v12;
      sub_15F1EA0((__int64)v12, *(_QWORD *)a3, 55, (__int64)(v12 - 9), 3, 0);
      if ( *(v9 - 9) )
      {
        v13 = *(v9 - 8);
        v14 = *(v9 - 7) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v14 = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
      }
      *(v9 - 9) = a2;
      v15 = *(_QWORD *)(a2 + 8);
      *(v9 - 8) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = (unsigned __int64)(v9 - 8) | *(_QWORD *)(v15 + 16) & 3LL;
      *(v9 - 7) = (a2 + 8) | *(v9 - 7) & 3LL;
      *(_QWORD *)(a2 + 8) = v35;
      if ( *(v9 - 6) )
      {
        v16 = *(v9 - 5);
        v17 = *(v9 - 4) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v17 = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
      }
      *(v9 - 6) = a3;
      v18 = *(_QWORD *)(a3 + 8);
      *(v9 - 5) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = (unsigned __int64)(v9 - 5) | *(_QWORD *)(v18 + 16) & 3LL;
      *(v9 - 4) = (a3 + 8) | *(v9 - 4) & 3LL;
      *(_QWORD *)(a3 + 8) = v9 - 6;
      if ( *(v9 - 3) )
      {
        v19 = *(v9 - 2);
        v20 = *(v9 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v20 = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
      }
      *(v9 - 3) = v34;
      if ( v34 )
      {
        v21 = *(_QWORD *)(v34 + 8);
        *(v9 - 2) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = (unsigned __int64)(v9 - 2) | *(_QWORD *)(v21 + 16) & 3LL;
        *(v9 - 1) = (v34 + 8) | *(v9 - 1) & 3LL;
        *(_QWORD *)(v34 + 8) = v9 - 3;
      }
      sub_164B780((__int64)v9, v39);
    }
    else
    {
      v38 = 0;
    }
    if ( a6 && (*(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0) )
    {
      v22 = sub_1625790(a6, 2);
      v23 = v22;
      if ( *(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0 )
      {
        v24 = sub_1625790(a6, 15);
        if ( v23 )
          sub_1625C10(v38, 2, v23);
        if ( v24 )
          sub_1625C10(v38, 15, v24);
      }
      else if ( v22 )
      {
        sub_1625C10(v38, 2, v22);
      }
    }
    v25 = *(_QWORD *)(a1 + 8);
    if ( v25 )
    {
      v26 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v25 + 40, (__int64)v9);
      v27 = v9[3];
      v28 = *v26;
      v9[4] = v26;
      v28 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v28 | v27 & 7;
      *(_QWORD *)(v28 + 8) = v9 + 3;
      *v26 = *v26 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780(v38, a5);
    v30 = *(_QWORD *)(a1 + 80) == 0;
    v39[0] = (__int64)v9;
    if ( v30 )
      sub_4263D6(v38, a5, v29);
    (*(void (__fastcall **)(__int64, __int64 *))(a1 + 88))(a1 + 64, v39);
    v31 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v39[0] = *(_QWORD *)a1;
      sub_1623A60((__int64)v39, v31, 2);
      v32 = v9[6];
      if ( v32 )
        sub_161E7C0((__int64)(v9 + 6), v32);
      v33 = (unsigned __int8 *)v39[0];
      v9[6] = v39[0];
      if ( v33 )
        sub_1623210((__int64)v39, v33, (__int64)(v9 + 6));
    }
  }
  else
  {
    v9 = (_QWORD *)sub_15A2DC0(a2, (__int64 *)a3, a4, 0);
    v10 = sub_14DBA30((__int64)v9, *(_QWORD *)(a1 + 96), 0);
    if ( v10 )
      return (_QWORD *)v10;
  }
  return v9;
}
