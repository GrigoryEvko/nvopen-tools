// Function: sub_8CE3E0
// Address: 0x8ce3e0
//
__int64 __fastcall sub_8CE3E0(__int64 a1)
{
  __int64 **v1; // rax
  __int64 *v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // r12d
  __int64 v6; // r12
  __int64 v7; // rcx
  __int64 v8; // r15
  char v9; // al
  __int64 v10; // r14
  __int64 v11; // r10
  char v12; // dl
  __int64 v13; // rcx
  int v14; // edx
  int v15; // ecx
  _QWORD **v16; // rdx
  __int64 **v17; // rcx
  _DWORD *v18; // r8
  __int64 v19; // rsi
  char v20; // al
  __int64 *v21; // rax
  __int64 v22; // rdi
  _QWORD *i; // rbx
  unsigned __int8 v24; // al
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  __int16 v31; // dx
  __int64 v32; // rax
  _QWORD *j; // rbx
  _QWORD *k; // rbx
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v1 = *(__int64 ***)(a1 + 32);
  if ( !v1 )
    return 1;
  v2 = *v1;
  v3 = a1;
  if ( (__int64 *)a1 == *v1 )
  {
    v3 = (__int64)v1[1];
    if ( !v3 || a1 == v3 )
      return 1;
    v4 = 0;
    if ( *(_BYTE *)(v3 + 120) != *((_BYTE *)v2 + 120) )
      return v4;
  }
  else
  {
    v4 = 0;
    if ( *(_BYTE *)(a1 + 120) != *((_BYTE *)v2 + 120) )
      return v4;
  }
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(v3 + 200);
  v8 = *v2;
  v9 = *(_BYTE *)(*(_QWORD *)a1 + 80LL);
  if ( (unsigned __int8)(v9 - 19) > 3u )
  {
    if ( ((*(_BYTE *)(v7 + 88) ^ *((_BYTE *)v2 + 88)) & 0x73) != 0 )
      goto LABEL_36;
    v11 = 0;
    v10 = 0;
    if ( v9 != 3 )
      goto LABEL_8;
    return 1;
  }
  if ( ((*(_BYTE *)(v7 + 88) ^ *((_BYTE *)v2 + 88)) & 0x73) != 0 )
    goto LABEL_36;
  v10 = *(_QWORD *)(v6 + 88);
  v11 = *(_QWORD *)(v8 + 88);
  if ( v9 == 19 )
  {
    v26 = *(_QWORD *)(v8 + 88);
    v25 = 0;
    if ( !v10 )
      goto LABEL_45;
    if ( ((*(_BYTE *)(v11 + 265) ^ *(_BYTE *)(v10 + 265)) & 1) != 0 )
      goto LABEL_36;
    v18 = (_DWORD *)(v6 + 48);
    v17 = *(__int64 ***)(v8 + 88);
    v16 = *(_QWORD ***)(v6 + 88);
    goto LABEL_64;
  }
LABEL_8:
  if ( dword_4F077C4 != 2 || (unsigned __int8)(v9 - 4) > 2u )
  {
    v12 = *(_BYTE *)(v7 + 121);
    v13 = v2[25];
    v14 = v12 & 1;
    v15 = v13 ? *(_BYTE *)(v13 + 121) & 1 : *((_BYTE *)v2 + 121) & 1;
    if ( v15 != v14 )
      goto LABEL_36;
  }
  if ( v10 )
  {
    v16 = *(_QWORD ***)(v6 + 88);
    v17 = *(__int64 ***)(v8 + 88);
    v18 = (_DWORD *)(v6 + 48);
    if ( v9 == 20 )
    {
      v19 = *v16[41];
LABEL_15:
      v20 = *(_BYTE *)(v8 + 80);
      if ( v20 == 20 )
      {
        v22 = *v17[41];
      }
      else
      {
        if ( v20 == 21 )
          v21 = v17[29];
        else
          v21 = v17[4];
        v22 = *v21;
      }
      v35 = v11;
      if ( !(unsigned int)sub_89B3C0(v22, v19, 0, 0, v18, 8u) )
        goto LABEL_36;
      v9 = *(_BYTE *)(v6 + 80);
      v11 = v35;
      if ( v9 != 19 )
        goto LABEL_21;
      if ( *(_QWORD *)(v3 + 200) == v3 )
      {
        v24 = *(_BYTE *)(v35 + 160);
        if ( ((v24 ^ *(_BYTE *)(v10 + 160)) & 1) != 0
          && ((*(_BYTE *)(v10 + 160) & 1) == 0 && (*(_BYTE *)(v10 + 266) & 8) != 0
           || (v24 & 1) == 0 && (*(_BYTE *)(v35 + 266) & 8) != 0) )
        {
LABEL_36:
          v4 = 0;
          sub_8C6700((__int64 *)v3, (unsigned int *)v2 + 16, 0x42Au, 0x425u);
          sub_8C7090(59, v3);
          return v4;
        }
      }
      goto LABEL_44;
    }
    if ( v9 == 21 )
    {
      v19 = *v16[29];
      goto LABEL_15;
    }
LABEL_64:
    v19 = *v16[4];
    goto LABEL_15;
  }
  if ( v9 != 19 )
  {
LABEL_21:
    if ( v9 == 20 )
    {
      for ( i = *(_QWORD **)(v10 + 168); i; i = (_QWORD *)*i )
        sub_8CDA30(*(_QWORD *)(i[3] + 88LL));
      sub_8CDA30(*(_QWORD *)(v10 + 176));
    }
    return 1;
  }
LABEL_44:
  v25 = *(_QWORD *)(v6 + 88);
  v26 = *(_QWORD *)(v8 + 88);
LABEL_45:
  v27 = *(_QWORD *)(v25 + 88);
  if ( v27 && (*(_BYTE *)(v25 + 160) & 1) == 0 )
    v25 = *(_QWORD *)(v27 + 88);
  v28 = *(_QWORD *)(v25 + 176);
  v29 = *(_QWORD *)(v26 + 88);
  if ( v29 && (*(_BYTE *)(v26 + 160) & 1) == 0 )
    v26 = *(_QWORD *)(v29 + 88);
  if ( *(_BYTE *)(v28 + 80) != 3 )
  {
    v30 = *(_QWORD *)(v26 + 176);
    v31 = 32;
    v32 = *(_QWORD *)(v30 + 88);
    if ( (*(_BYTE *)(v10 + 160) & 8) == 0 )
      v31 = 32 * ((*(_BYTE *)(v11 + 160) & 8) != 0);
    v36 = *(_QWORD *)(v28 + 88);
    v4 = sub_89AB40(
           *(_QWORD *)(*(_QWORD *)(v36 + 168) + 168LL),
           *(_QWORD *)(*(_QWORD *)(v32 + 168) + 168LL),
           v31,
           v27,
           (_UNKNOWN *__ptr32 *)v36);
    if ( !v4 )
    {
      sub_8C6700((__int64 *)v3, (unsigned int *)v2 + 16, 0x42Au, 0x425u);
      sub_8C7090(59, v3);
      return v4;
    }
    if ( (*(_BYTE *)(v3 + 121) & 8) != 0 && (*(_BYTE *)(v28 + 81) & 2) != 0 )
      return 1;
    v4 = sub_8CD5A0((__int64 *)v36);
    if ( v4 )
    {
      for ( j = *(_QWORD **)(v10 + 168); j; j = (_QWORD *)*j )
        sub_8CD5A0(*(__int64 **)(j[1] + 88LL));
      return v4;
    }
    return 0;
  }
  v4 = sub_8CD5A0(*(__int64 **)(v28 + 88));
  if ( !v4 )
    return 0;
  for ( k = *(_QWORD **)(v10 + 168); k; k = (_QWORD *)*k )
    sub_8CD5A0(*(__int64 **)(k[1] + 88LL));
  return v4;
}
