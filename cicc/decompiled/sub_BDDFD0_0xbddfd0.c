// Function: sub_BDDFD0
// Address: 0xbddfd0
//
unsigned __int64 __fastcall sub_BDDFD0(__int64 a1, const char *a2)
{
  const char *v3; // rbx
  unsigned __int8 v4; // al
  __int64 v5; // rsi
  unsigned __int64 result; // rax
  const char *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r12
  _BYTE *v10; // rax
  unsigned __int8 v11; // al
  const char **v12; // rbx
  const char *v13; // r14
  int v14; // r15d
  int v15; // ebx
  char v16; // dl
  __int64 v17; // r15
  _BYTE *v18; // rax
  char v19; // dl
  __int64 v20; // r15
  _BYTE *v21; // rax
  __int64 v22; // rdx
  char v23; // al
  char v24; // al
  __int64 v25; // r15
  _BYTE *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp+10h] [rbp-80h] BYREF
  __int64 v29; // [rsp+18h] [rbp-78h]
  char v30; // [rsp+20h] [rbp-70h]
  _QWORD v31[4]; // [rsp+30h] [rbp-60h] BYREF
  char v32; // [rsp+50h] [rbp-40h]
  char v33; // [rsp+51h] [rbp-3Fh]

  v3 = a2 - 16;
  v4 = *(a2 - 16);
  if ( (v4 & 2) == 0 )
  {
    v5 = *(_QWORD *)&v3[-8 * ((v4 >> 2) & 0xF)];
    if ( v5 )
      goto LABEL_3;
LABEL_11:
    v9 = *(_QWORD *)a1;
    v33 = 1;
    v31[0] = "missing variable";
    v32 = 3;
    if ( v9 )
    {
      sub_CA0E80(v31, v9);
      v10 = *(_BYTE **)(v9 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
      {
        sub_CB5D20(v9, 10);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = v10 + 1;
        *v10 = 10;
      }
    }
    goto LABEL_14;
  }
  v5 = **((_QWORD **)a2 - 4);
  if ( !v5 )
    goto LABEL_11;
LABEL_3:
  sub_BDDCF0(a1, v5);
  result = *((unsigned __int8 *)a2 - 16);
  if ( (result & 2) != 0 )
  {
    v7 = (const char *)*((_QWORD *)a2 - 4);
  }
  else
  {
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v7 = &v3[-result];
  }
  v8 = *((_QWORD *)v7 + 1);
  if ( !v8 )
    return result;
  if ( !sub_AF4230(*((_QWORD *)v7 + 1)) )
  {
    v20 = *(_QWORD *)a1;
    v33 = 1;
    v31[0] = "invalid expression";
    v32 = 3;
    if ( v20 )
    {
      sub_CA0E80(v31, v20);
      v21 = *(_BYTE **)(v20 + 32);
      if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 24) )
      {
        sub_CB5D20(v20, 10);
      }
      else
      {
        *(_QWORD *)(v20 + 32) = v21 + 1;
        *v21 = 10;
      }
      v22 = *(_QWORD *)a1;
      v23 = *(_BYTE *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= v23;
      if ( v22 )
        sub_BD9900((__int64 *)a1, (const char *)v8);
    }
    else
    {
      v24 = *(_BYTE *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= v24;
    }
  }
  result = sub_AF47B0((__int64)&v28, *(unsigned __int64 **)(v8 + 16), *(unsigned __int64 **)(v8 + 24));
  if ( !v30 )
    return result;
  v11 = *(a2 - 16);
  v12 = (v11 & 2) != 0 ? (const char **)*((_QWORD *)a2 - 4) : (const char **)&v3[-8 * ((v11 >> 2) & 0xF)];
  v13 = *v12;
  v14 = v29;
  v15 = v28;
  result = sub_AF3FE0((__int64)v13);
  if ( !v16 )
    return result;
  if ( result < (unsigned int)(v15 + v14) )
  {
    v25 = *(_QWORD *)a1;
    v33 = 1;
    v31[0] = "fragment is larger than or outside of variable";
    v32 = 3;
    if ( v25 )
    {
      sub_CA0E80(v31, v25);
      v26 = *(_BYTE **)(v25 + 32);
      if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 24) )
      {
        sub_CB5D20(v25, 10);
      }
      else
      {
        *(_QWORD *)(v25 + 32) = v26 + 1;
        *v26 = 10;
      }
      v27 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= result;
      if ( v27 )
        goto LABEL_24;
      return result;
    }
LABEL_14:
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
    return result;
  }
  if ( result != v15 )
    return result;
  v17 = *(_QWORD *)a1;
  v33 = 1;
  v31[0] = "fragment covers entire variable";
  v32 = 3;
  if ( !v17 )
    goto LABEL_14;
  sub_CA0E80(v31, v17);
  v18 = *(_BYTE **)(v17 + 32);
  if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
  {
    sub_CB5D20(v17, 10);
  }
  else
  {
    *(_QWORD *)(v17 + 32) = v18 + 1;
    *v18 = 10;
  }
  result = *(_QWORD *)a1;
  v19 = *(_BYTE *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= v19;
  if ( result )
  {
LABEL_24:
    result = (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
    if ( v13 )
      return (unsigned __int64)sub_BD9900((__int64 *)a1, v13);
  }
  return result;
}
