// Function: sub_AF5710
// Address: 0xaf5710
//
bool __fastcall sub_AF5710(__int64 *a1, __int64 a2)
{
  unsigned __int8 v3; // r14
  __int64 v4; // rdx
  _QWORD *v5; // r15
  __int64 v7; // rax
  _BYTE *v8; // r8
  __int64 v9; // rsi
  _BYTE *v10; // rax
  unsigned int v11; // r9d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned int v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  _BYTE *v25; // [rsp+18h] [rbp-38h]
  _BYTE *v26; // [rsp+18h] [rbp-38h]

  v3 = *(_BYTE *)(a2 - 16);
  v4 = *a1;
  if ( (v3 & 2) != 0 )
  {
    v5 = *(_QWORD **)(a2 - 32);
    if ( v4 != v5[1] )
      return 0;
  }
  else
  {
    v5 = (_QWORD *)(a2 - 16 - 8LL * ((v3 >> 2) & 0xF));
    if ( v4 != v5[1] )
      return 0;
  }
  v25 = (_BYTE *)(a2 - 16);
  v21 = a2;
  if ( a1[1] != sub_AF5140(a2, 2u) )
    return 0;
  v7 = sub_AF5140(a2, 3u);
  v8 = (_BYTE *)(a2 - 16);
  if ( a1[2] != v7 )
    return 0;
  v9 = a1[3];
  if ( *(_BYTE *)a2 != 16 )
  {
    v19 = a1[3];
    v10 = sub_A17150(v25);
    v9 = v19;
    v8 = v25;
    v21 = *(_QWORD *)v10;
  }
  if ( v9 != v21 )
    return 0;
  if ( *((_DWORD *)a1 + 8) != *(_DWORD *)(a2 + 16) )
    return 0;
  v26 = v8;
  v22 = a1[5];
  if ( v22 != *((_QWORD *)sub_A17150(v8) + 4) || *((_DWORD *)a1 + 12) != *(_DWORD *)(a2 + 20) )
    return 0;
  if ( (v3 & 2) == 0 )
  {
    v12 = 0;
    v11 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    if ( v11 <= 8 )
      goto LABEL_16;
    goto LABEL_15;
  }
  v11 = *(_DWORD *)(a2 - 24);
  v12 = 0;
  if ( v11 > 8 )
LABEL_15:
    v12 = v5[8];
LABEL_16:
  if ( a1[7] != v12
    || *((_DWORD *)a1 + 16) != *(_DWORD *)(a2 + 24)
    || *((_DWORD *)a1 + 17) != *(_DWORD *)(a2 + 28)
    || *((_DWORD *)a1 + 18) != *(_DWORD *)(a2 + 32)
    || *((_DWORD *)a1 + 19) != *(_DWORD *)(a2 + 36) )
  {
    return 0;
  }
  v13 = a1[10];
  if ( (v3 & 2) != 0 )
  {
    if ( v13 != *(_QWORD *)(*(_QWORD *)(a2 - 32) + 40LL) )
      return 0;
  }
  else if ( v13 != *(_QWORD *)&v26[-8 * ((v3 >> 2) & 0xF) + 40] )
  {
    return 0;
  }
  v14 = 0;
  if ( v11 > 9 )
    v14 = v5[9];
  if ( v14 != a1[11] )
    return 0;
  v20 = v11;
  v23 = a1[12];
  if ( v23 != *((_QWORD *)sub_A17150(v26) + 6) )
    return 0;
  v24 = a1[13];
  if ( v24 != *((_QWORD *)sub_A17150(v26) + 7) )
    return 0;
  v15 = a1[14];
  if ( (v3 & 2) != 0 )
  {
    if ( v20 > 0xA )
    {
      if ( v15 != v5[10] )
        return 0;
      v16 = a1[15];
      if ( v20 != 11 )
      {
        v17 = *(_BYTE **)(a2 - 32);
        if ( v16 != *((_QWORD *)v17 + 11) )
          return 0;
        goto LABEL_33;
      }
      goto LABEL_43;
    }
LABEL_41:
    if ( v15 )
      return 0;
    v16 = a1[15];
    goto LABEL_43;
  }
  if ( v20 <= 0xA )
    goto LABEL_41;
  if ( v15 != v5[10] )
    return 0;
  v16 = a1[15];
  if ( v20 != 11 )
  {
    v17 = &v26[-8 * ((v3 >> 2) & 0xF)];
    if ( v16 != *((_QWORD *)v17 + 11) )
      return 0;
LABEL_33:
    v18 = a1[16];
    if ( v20 > 0xC )
      v16 = *((_QWORD *)v17 + 12);
    else
      v16 = 0;
    return v16 == v18;
  }
LABEL_43:
  if ( v16 )
    return 0;
  v18 = a1[16];
  return v16 == v18;
}
