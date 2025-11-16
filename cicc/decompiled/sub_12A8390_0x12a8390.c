// Function: sub_12A8390
// Address: 0x12a8390
//
__int64 __fastcall sub_12A8390(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  char i; // al
  unsigned __int64 v8; // r15
  __int64 v9; // rbx
  char *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rdi
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rsi
  _QWORD *v31; // [rsp+10h] [rbp-90h]
  unsigned __int64 *v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v34[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v35; // [rsp+40h] [rbp-60h]
  _QWORD v36[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v37; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(*(_QWORD *)(a3 + 72) + 16LL);
  v6 = **(_QWORD **)(v5 + 16);
  for ( i = *(_BYTE *)(v6 + 140); i == 12; i = *(_BYTE *)(v6 + 140) )
    v6 = *(_QWORD *)(v6 + 160);
  if ( i != 6 )
    sub_127B550("expected va_arg builtin second argument tobe of pointer type", (_DWORD *)(a3 + 36), 1);
  v8 = *(_QWORD *)(v6 + 160);
  v9 = sub_127A030(a2[4] + 8LL, v8, 0);
  v10 = sub_128F980((__int64)a2, v5);
  v31 = sub_12812E0(a2, (__int64)v10, v9);
  v36[0] = "varg_temp";
  v37 = 259;
  v11 = sub_127FE40(a2, v8, (__int64)v36);
  v37 = 257;
  v12 = (__int64)v11;
  v13 = sub_1648A60(64, 2);
  v14 = (_QWORD *)v13;
  if ( v13 )
    sub_15F9650(v13, v31, v12, 0, 0);
  v15 = a2[7];
  if ( v15 )
  {
    v32 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v15 + 40, v14);
    v16 = *v32;
    v17 = v14[3] & 7LL;
    v14[4] = v32;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v16 | v17;
    *(_QWORD *)(v16 + 8) = v14 + 3;
    *v32 = *v32 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780(v14, v36);
  v18 = a2[6];
  if ( v18 )
  {
    v34[0] = a2[6];
    sub_1623A60(v34, v18, 2);
    v19 = v14 + 6;
    if ( v14[6] )
    {
      sub_161E7C0(v14 + 6);
      v19 = v14 + 6;
    }
    v20 = v34[0];
    v14[6] = v34[0];
    if ( v20 )
      sub_1623210(v34, v20, v19);
  }
  if ( *(char *)(v8 + 142) >= 0 && *(_BYTE *)(v8 + 140) == 12 )
    v21 = (unsigned int)sub_8D4AB0(v8);
  else
    v21 = *(unsigned int *)(v8 + 136);
  sub_15F9450(v14, v21);
  v22 = a2[4];
  v35 = 257;
  v23 = *(_QWORD *)(v22 + 728);
  if ( v23 != *(_QWORD *)v12 )
  {
    if ( *(_BYTE *)(v12 + 16) > 0x10u )
    {
      v37 = 257;
      v12 = sub_15FDBD0(47, v12, v23, v36, 0);
      v25 = a2[7];
      if ( v25 )
      {
        v26 = (__int64 *)a2[8];
        sub_157E9D0(v25 + 40, v12);
        v27 = *(_QWORD *)(v12 + 24);
        v28 = *v26;
        *(_QWORD *)(v12 + 32) = v26;
        v28 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v12 + 24) = v28 | v27 & 7;
        *(_QWORD *)(v28 + 8) = v12 + 24;
        *v26 = *v26 & 7 | (v12 + 24);
      }
      sub_164B780(v12, v34);
      v29 = a2[6];
      if ( v29 )
      {
        v33 = a2[6];
        sub_1623A60(&v33, v29, 2);
        if ( *(_QWORD *)(v12 + 48) )
          sub_161E7C0(v12 + 48);
        v30 = v33;
        *(_QWORD *)(v12 + 48) = v33;
        if ( v30 )
          sub_1623210(&v33, v30, v12 + 48);
      }
    }
    else
    {
      v12 = sub_15A46C0(47, v12, v23, 0);
    }
  }
  *(_QWORD *)a1 = v12;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
