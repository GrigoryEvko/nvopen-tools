// Function: sub_13D05E0
// Address: 0x13d05e0
//
__int64 __fastcall sub_13D05E0(unsigned __int8 *a1, unsigned __int8 *a2, char a3, _QWORD *a4)
{
  __int64 v5; // r12
  unsigned __int8 *v6; // rbx
  __int64 v7; // r13
  unsigned __int8 *v9; // rdi
  unsigned __int8 v10; // al
  char v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-88h]
  __int64 v20; // [rsp+10h] [rbp-80h]
  __int64 v21; // [rsp+10h] [rbp-80h]
  __int64 v22; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+18h] [rbp-78h]
  __int64 v24; // [rsp+18h] [rbp-78h]
  _BYTE v25[32]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v26[8]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v27; // [rsp+48h] [rbp-48h] BYREF
  __int64 v28; // [rsp+50h] [rbp-40h]

  v5 = (__int64)a2;
  v6 = a1;
  if ( a1[16] <= 0x10u )
  {
    if ( a2[16] > 0x10u )
    {
      v6 = a2;
      v5 = (__int64)a1;
    }
    else
    {
      v7 = sub_14D6F90(16, a1, a2, *a4);
      if ( v7 )
        return v7;
      if ( a1[16] == 9 )
        goto LABEL_7;
    }
  }
  if ( *(_BYTE *)(v5 + 16) == 9 )
  {
LABEL_7:
    v9 = *(unsigned __int8 **)v6;
    v7 = sub_15A11D0(*(_QWORD *)v6, 0, 0);
    goto LABEL_8;
  }
  v9 = v6;
  v7 = sub_13CDA40(v6, (_QWORD *)v5);
LABEL_8:
  if ( v7 )
    return v7;
  v10 = *(_BYTE *)(v5 + 16);
  if ( v10 == 14 )
  {
    v20 = sub_1698280(v9);
    sub_169D3F0(v25, 1.0);
    sub_169E320(&v27, v25, v20);
    sub_1698460(v25);
    sub_16A3360(v26, *(_QWORD *)(v5 + 32), 0, v25);
    v11 = sub_1594120(v5, v26);
    if ( v27 != sub_16982C0() )
    {
LABEL_11:
      sub_1698460(&v27);
      goto LABEL_12;
    }
    v14 = v28;
    if ( v28 )
    {
      v15 = v28 + 32LL * *(_QWORD *)(v28 - 8);
      if ( v28 != v15 )
      {
        do
        {
          v21 = v14;
          v23 = v15 - 32;
          sub_127D120((_QWORD *)(v15 - 24));
          v15 = v23;
          v14 = v21;
        }
        while ( v21 != v23 );
      }
LABEL_32:
      j_j_j___libc_free_0_0(v14 - 8);
    }
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
      goto LABEL_18;
    if ( v10 > 0x10u )
      goto LABEL_18;
    v16 = sub_15A1020(v5);
    v17 = v16;
    if ( !v16 || *(_BYTE *)(v16 + 16) != 14 )
      goto LABEL_18;
    v19 = sub_1698280(v5);
    sub_169D3F0(v25, 1.0);
    sub_169E320(&v27, v25, v19);
    sub_1698460(v25);
    sub_16A3360(v26, *(_QWORD *)(v17 + 32), 0, v25);
    v11 = sub_1594120(v17, v26);
    if ( v27 != sub_16982C0() )
      goto LABEL_11;
    v14 = v28;
    if ( v28 )
    {
      v18 = v28 + 32LL * *(_QWORD *)(v28 - 8);
      if ( v28 != v18 )
      {
        do
        {
          v22 = v14;
          v24 = v18 - 32;
          sub_127D120((_QWORD *)(v18 - 24));
          v18 = v24;
          v14 = v22;
        }
        while ( v22 != v24 );
      }
      goto LABEL_32;
    }
  }
LABEL_12:
  if ( v11 )
    return (__int64)v6;
LABEL_18:
  if ( (a3 & 0xA) != 0xA || !(unsigned __int8)sub_13CC1F0(v5) )
  {
    if ( v6 == (unsigned __int8 *)v5 && v6[16] == 78 )
    {
      v12 = *((_QWORD *)v6 - 3);
      if ( !*(_BYTE *)(v12 + 16) && *(_DWORD *)(v12 + 36) == 196 )
      {
        v13 = ((unsigned __int64)v6 & 0xFFFFFFFFFFFFFFF8LL)
            - 24LL * (*(_DWORD *)(((unsigned __int64)v6 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
        if ( *(_QWORD *)v13 )
        {
          if ( (a3 & 2) != 0 && (a3 & 1) != 0 && (a3 & 8) != 0 )
            return *(_QWORD *)v13;
        }
      }
    }
    return v7;
  }
  return sub_15A06D0(*(_QWORD *)v6);
}
