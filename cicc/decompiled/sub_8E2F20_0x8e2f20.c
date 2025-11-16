// Function: sub_8E2F20
// Address: 0x8e2f20
//
__int64 __fastcall sub_8E2F20(
        __int64 a1,
        int a2,
        int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        int a8,
        int *a9)
{
  __int64 v9; // r13
  __int64 v10; // r12
  char i; // al
  __int64 v12; // r8
  _BOOL4 v13; // r15d
  int v14; // eax
  unsigned int v15; // r9d
  int v17; // eax
  int v18; // eax
  _BOOL4 v19; // eax
  _BOOL4 v20; // eax
  __int64 v21; // [rsp+0h] [rbp-70h]
  unsigned int v22; // [rsp+Ch] [rbp-64h]
  int v23; // [rsp+10h] [rbp-60h]
  int v24; // [rsp+2Ch] [rbp-44h] BYREF
  int v25; // [rsp+30h] [rbp-40h] BYREF
  int v26; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v27[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = a1;
  v10 = a6;
  *a7 = 0;
  *a9 = 0;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    v9 = *(_QWORD *)(v9 + 160);
  while ( *(_BYTE *)(v9 + 140) == 12 );
  for ( i = *(_BYTE *)(a6 + 140); i == 12; i = *(_BYTE *)(v10 + 140) )
  {
    v10 = *(_QWORD *)(v10 + 160);
LABEL_5:
    ;
  }
  if ( (*(_BYTE *)(v10 + 141) & 0x20) != 0 && i != 1 )
  {
    v21 = a5;
    v22 = a4;
    v23 = a3;
    v18 = sub_8D2690(v10);
    a3 = v23;
    a4 = v22;
    a5 = v21;
    if ( !v18 )
      return 0;
  }
  v13 = sub_8E2840(v9, a2, a3, a4, a5, v10, 1, a8, &v24, &v25);
  v14 = v24;
  if ( v24 == 1373 )
  {
    v24 = 1374;
    if ( v13 )
    {
      v14 = 1374;
      goto LABEL_23;
    }
LABEL_14:
    if ( dword_4F077C4 == 2 )
      goto LABEL_28;
LABEL_15:
    if ( sub_8DB0D0(v9, v10, &v26, v27, v12) )
    {
      v17 = v26;
      if ( !v26 || LODWORD(v27[0]) )
      {
        *a7 = 1;
        v15 = 1;
        *a9 = v17;
        return v15;
      }
      if ( !v13 )
      {
        *a9 = v26;
        v15 = 1;
        *a7 = 1;
        return v15;
      }
    }
    else if ( !v13 )
    {
      return 0;
    }
    v14 = v24;
    goto LABEL_11;
  }
  if ( !v13 )
    goto LABEL_14;
  if ( !v24 )
  {
LABEL_11:
    *a9 = v14;
    return 1;
  }
LABEL_23:
  if ( v25 )
    goto LABEL_11;
  if ( dword_4F077C4 != 2 )
    goto LABEL_15;
LABEL_28:
  v19 = sub_8DEFB0(v9, v10, 1, 0);
  v15 = 1;
  if ( !v19 )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_15;
    if ( !(unsigned int)sub_8D2F30(v9, v10) )
      goto LABEL_15;
    v20 = sub_8D5EF0(v9, v10, &v26, v27);
    v15 = 1;
    if ( !v20 )
      goto LABEL_15;
  }
  return v15;
}
