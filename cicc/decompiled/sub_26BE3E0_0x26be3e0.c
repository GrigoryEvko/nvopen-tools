// Function: sub_26BE3E0
// Address: 0x26be3e0
//
__int64 __fastcall sub_26BE3E0(_QWORD *a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r15
  __int64 result; // rax
  _QWORD *v4; // rbx
  char v5; // dl
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r8
  bool v11; // cf
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // r8
  __int64 v15; // r13
  unsigned __int64 v16; // r13
  int *v17; // r13
  size_t v18; // r12
  size_t v19; // r13
  __int64 v20; // [rsp+10h] [rbp-F0h]
  char v21; // [rsp+10h] [rbp-F0h]
  __int64 v22; // [rsp+18h] [rbp-E8h]
  __int64 v23; // [rsp+18h] [rbp-E8h]
  int *v24; // [rsp+18h] [rbp-E8h]
  _QWORD v25[2]; // [rsp+20h] [rbp-E0h] BYREF
  int v26[52]; // [rsp+30h] [rbp-D0h] BYREF

  v1 = a1;
  v2 = (_QWORD *)*a1;
  while ( 1 )
  {
    result = (__int64)&unk_4F838D3;
    v4 = (_QWORD *)*(v1 - 1);
    v5 = unk_4F838D3;
    if ( unk_4F838D3 )
    {
      v6 = v2[8];
      if ( v6 )
        break;
    }
    result = v2[20];
    if ( v2[14] )
    {
      v12 = v2[12];
      if ( !result
        || (v13 = v2[18], result = *(unsigned int *)(v13 + 32), *(_DWORD *)(v12 + 32) < (unsigned int)result)
        || *(_DWORD *)(v12 + 32) == (_DWORD)result
        && (result = *(unsigned int *)(v13 + 36), *(_DWORD *)(v12 + 36) < (unsigned int)result) )
      {
        v6 = *(_QWORD *)(v12 + 40);
        goto LABEL_23;
      }
    }
    else
    {
      if ( !result )
        goto LABEL_35;
      v13 = v2[18];
    }
    v14 = *(_QWORD *)(v13 + 64);
    v15 = v13 + 48;
    if ( v14 != v15 )
    {
      v6 = 0;
      do
      {
        v21 = v5;
        v23 = v14;
        v6 += sub_EF9210((_QWORD *)(v14 + 48));
        result = sub_220EF30(v23);
        v5 = v21;
        v14 = result;
      }
      while ( v15 != result );
LABEL_23:
      if ( v6 )
        goto LABEL_24;
    }
LABEL_35:
    v6 = v2[7] != 0;
LABEL_24:
    if ( v5 )
      break;
    result = v4[20];
    if ( !v4[14] )
      goto LABEL_26;
LABEL_6:
    v8 = v4[12];
    if ( !result
      || (v9 = v4[18], result = *(unsigned int *)(v9 + 32), *(_DWORD *)(v8 + 32) < (unsigned int)result)
      || *(_DWORD *)(v8 + 32) == (_DWORD)result
      && (result = *(unsigned int *)(v9 + 36), *(_DWORD *)(v8 + 36) < (unsigned int)result) )
    {
      v7 = *(_QWORD *)(v8 + 40);
      goto LABEL_12;
    }
LABEL_9:
    v10 = *(_QWORD *)(v9 + 64);
    result = v9 + 48;
    v20 = v9 + 48;
    if ( v10 == v9 + 48 )
      goto LABEL_27;
    v7 = 0;
    do
    {
      v22 = v10;
      v7 += sub_EF9210((_QWORD *)(v10 + 48));
      result = sub_220EF30(v22);
      v10 = result;
    }
    while ( v20 != result );
LABEL_12:
    if ( !v7 )
      goto LABEL_27;
LABEL_13:
    v11 = v7 < v6;
    if ( v7 == v6 )
      goto LABEL_28;
LABEL_14:
    if ( !v11 )
      goto LABEL_42;
LABEL_15:
    *v1-- = v4;
  }
  v7 = v4[8];
  if ( v7 )
    goto LABEL_13;
  result = v4[20];
  if ( v4[14] )
    goto LABEL_6;
LABEL_26:
  if ( result )
  {
    v9 = v4[18];
    goto LABEL_9;
  }
LABEL_27:
  v16 = v4[7] != 0;
  v11 = v16 < v6;
  if ( v16 != v6 )
    goto LABEL_14;
LABEL_28:
  v17 = (int *)v2[2];
  v18 = v2[3];
  if ( v17 )
  {
    sub_C7D030(v26);
    sub_C7D280(v26, v17, v18);
    result = sub_C7D290(v26, v25);
    v18 = v25[0];
  }
  v19 = v4[3];
  v24 = (int *)v4[2];
  if ( v24 )
  {
    sub_C7D030(v26);
    sub_C7D280(v26, v24, v19);
    result = sub_C7D290(v26, v25);
    v19 = v25[0];
  }
  if ( v19 > v18 )
  {
    v4 = (_QWORD *)*(v1 - 1);
    goto LABEL_15;
  }
LABEL_42:
  *v1 = v2;
  return result;
}
