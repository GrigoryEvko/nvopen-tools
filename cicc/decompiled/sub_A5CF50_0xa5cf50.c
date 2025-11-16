// Function: sub_A5CF50
// Address: 0xa5cf50
//
__int64 __fastcall sub_A5CF50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned __int8 v5; // al
  __int64 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rcx
  unsigned __int8 v12; // al
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 *v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned __int8 v20; // al
  __int64 v21; // rcx
  __int64 v23; // rax
  __int64 *v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 *v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // [rsp+0h] [rbp-40h] BYREF
  char v32; // [rsp+8h] [rbp-38h]
  char *v33; // [rsp+10h] [rbp-30h]
  __int64 v34; // [rsp+18h] [rbp-28h]

  sub_904010(a1, "!DISubrange(");
  v34 = a3;
  v4 = a2 - 16;
  v33 = ", ";
  v5 = *(_BYTE *)(a2 - 16);
  v31 = a1;
  v32 = 1;
  if ( (v5 & 2) != 0 )
    v6 = *(__int64 **)(a2 - 32);
  else
    v6 = (__int64 *)(v4 - 8LL * ((v5 >> 2) & 0xF));
  v7 = *v6;
  if ( *v6 && *(_BYTE *)v7 == 1 )
  {
    v8 = *(_QWORD *)(v7 + 136);
    v9 = *(__int64 **)(v8 + 24);
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 > 0x40 )
    {
      v11 = *v9;
    }
    else
    {
      v11 = 0;
      if ( v10 )
        v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
    }
    sub_A538D0((__int64)&v31, "count", 5u, v11);
    v12 = *(_BYTE *)(a2 - 16);
    if ( (v12 & 2) == 0 )
    {
LABEL_9:
      v13 = *(_QWORD *)(v4 - 8LL * ((v12 >> 2) & 0xF) + 8);
      if ( !v13 )
        goto LABEL_10;
      goto LABEL_22;
    }
  }
  else
  {
    sub_A5CC00((__int64)&v31, "count", 5u, v7, 1);
    v12 = *(_BYTE *)(a2 - 16);
    if ( (v12 & 2) == 0 )
      goto LABEL_9;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( !v13 )
    goto LABEL_10;
LABEL_22:
  if ( *(_BYTE *)v13 != 1 )
  {
LABEL_10:
    sub_A5CC00((__int64)&v31, "lowerBound", 0xAu, v13, 1);
    v14 = *(_BYTE *)(a2 - 16);
    if ( (v14 & 2) != 0 )
      goto LABEL_11;
    goto LABEL_27;
  }
  v23 = *(_QWORD *)(v13 + 136);
  v24 = *(__int64 **)(v23 + 24);
  v25 = *(_DWORD *)(v23 + 32);
  if ( v25 > 0x40 )
  {
    v26 = *v24;
  }
  else
  {
    v26 = 0;
    if ( v25 )
      v26 = (__int64)((_QWORD)v24 << (64 - (unsigned __int8)v25)) >> (64 - (unsigned __int8)v25);
  }
  sub_A538D0((__int64)&v31, "lowerBound", 0xAu, v26);
  v14 = *(_BYTE *)(a2 - 16);
  if ( (v14 & 2) != 0 )
  {
LABEL_11:
    v15 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( !v15 )
      goto LABEL_28;
    goto LABEL_12;
  }
LABEL_27:
  v15 = *(_QWORD *)(a2 - 8LL * ((v14 >> 2) & 0xF));
  if ( !v15 )
  {
LABEL_28:
    sub_A5CC00((__int64)&v31, "upperBound", 0xAu, v15, 1);
    v20 = *(_BYTE *)(a2 - 16);
    if ( (v20 & 2) == 0 )
      goto LABEL_17;
LABEL_29:
    v21 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    if ( !v21 )
      goto LABEL_18;
    goto LABEL_30;
  }
LABEL_12:
  if ( *(_BYTE *)v15 != 1 )
    goto LABEL_28;
  v16 = *(_QWORD *)(v15 + 136);
  v17 = *(__int64 **)(v16 + 24);
  v18 = *(_DWORD *)(v16 + 32);
  if ( v18 > 0x40 )
  {
    v19 = *v17;
  }
  else
  {
    v19 = 0;
    if ( v18 )
      v19 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v18)) >> (64 - (unsigned __int8)v18);
  }
  sub_A538D0((__int64)&v31, "upperBound", 0xAu, v19);
  v20 = *(_BYTE *)(a2 - 16);
  if ( (v20 & 2) != 0 )
    goto LABEL_29;
LABEL_17:
  v21 = *(_QWORD *)(v4 - 8LL * ((v20 >> 2) & 0xF) + 24);
  if ( !v21 )
  {
LABEL_18:
    sub_A5CC00((__int64)&v31, "stride", 6u, v21, 1);
    return sub_904010(a1, ")");
  }
LABEL_30:
  if ( *(_BYTE *)v21 != 1 )
    goto LABEL_18;
  v27 = *(_QWORD *)(v21 + 136);
  v28 = *(__int64 **)(v27 + 24);
  v29 = *(_DWORD *)(v27 + 32);
  if ( v29 > 0x40 )
  {
    v30 = *v28;
  }
  else
  {
    v30 = 0;
    if ( v29 )
      v30 = (__int64)((_QWORD)v28 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
  }
  sub_A538D0((__int64)&v31, "stride", 6u, v30);
  return sub_904010(a1, ")");
}
