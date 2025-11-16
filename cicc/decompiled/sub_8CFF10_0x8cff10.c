// Function: sub_8CFF10
// Address: 0x8cff10
//
__int64 __fastcall sub_8CFF10(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // cl
  __int64 *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  char v9; // al
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  char v14; // al
  _QWORD *v15; // rdx
  __int64 v16; // r14
  __int64 v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+20h] [rbp-50h]
  char v26; // [rsp+2Fh] [rbp-41h]
  char v27; // [rsp+3Eh] [rbp-32h] BYREF
  char v28[49]; // [rsp+3Fh] [rbp-31h] BYREF

  v2 = *(_BYTE *)(a1 + 80);
  v26 = 0;
  if ( v2 <= 0x14u )
    v26 = (0x120C00uLL >> v2) & 1;
  v3 = *(__int64 **)(a1 + 64);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
  {
    if ( v3 )
    {
      v4 = sub_8CFEE0(*v3, a2);
      if ( v4 )
      {
        v24 = 0;
        v5 = *(_QWORD *)(v4 + 88);
        goto LABEL_8;
      }
    }
    goto LABEL_7;
  }
  v11 = sub_8CFEE0(*v3, a2);
  v12 = v11;
  if ( !v11 )
  {
LABEL_7:
    v24 = 0;
    v5 = 0;
    goto LABEL_8;
  }
  v24 = *(_QWORD *)(v11 + 88);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v24) && (unsigned int)sub_8D3A70(v24) )
    sub_8AD220(v24, 0);
  if ( v26 )
  {
    v14 = sub_877F80(a1);
    v15 = *(_QWORD **)(v12 + 96);
    if ( v14 == 2 )
    {
      v7 = v15[3];
      goto LABEL_27;
    }
    if ( v14 != 3 )
    {
      if ( v14 == 1 )
      {
        v7 = v15[1];
        goto LABEL_27;
      }
      v6 = sub_87D510(a1, &v27);
      v5 = *(_QWORD *)(v6 + 32);
      if ( !v5 )
      {
        v25 = 0;
        goto LABEL_9;
      }
LABEL_33:
      v25 = v5;
      v5 = 0;
      goto LABEL_34;
    }
    v19 = (_QWORD *)v15[6];
    if ( *(_BYTE *)(a1 + 80) != 20 )
      v19 = (_QWORD *)v15[5];
    v20 = sub_87D510(a1, &v27);
    v5 = *(_QWORD *)(v20 + 32);
    v21 = v20;
    if ( v5 )
    {
      if ( !v19 )
        goto LABEL_33;
    }
    else
    {
      sub_8C7090(v27, v20);
      v25 = *(_QWORD *)(v21 + 32);
      if ( !v19 )
        goto LABEL_34;
      v5 = *(_QWORD *)(v21 + 32);
    }
    do
    {
      v16 = v19[1];
      v22 = sub_87D1A0(v16, v28);
      if ( v22 && v5 == *(_QWORD *)(v22 + 32) && (unsigned int)sub_880F30(v16, a2) )
        return v16;
      v19 = (_QWORD *)*v19;
    }
    while ( v19 );
    return 0;
  }
  v5 = 0;
  if ( *(_BYTE *)(a1 + 80) != 19 || (v13 = *(_QWORD *)(a1 + 88), (v5 = *(_QWORD *)(v13 + 152)) == 0) )
  {
LABEL_8:
    v6 = sub_87D510(a1, &v27);
    v25 = *(_QWORD *)(v6 + 32);
    if ( v25 )
      goto LABEL_34;
    goto LABEL_9;
  }
  v7 = *(_QWORD *)(*(_QWORD *)(sub_8CFF10(*(_QWORD *)(v13 + 152), a2) + 88) + 144LL);
LABEL_27:
  v5 = 0;
  v6 = sub_87D510(a1, &v27);
  v25 = *(_QWORD *)(v6 + 32);
  if ( !v25 )
  {
    v25 = v7;
LABEL_9:
    v23 = v6;
    sub_8C7090(v27, v6);
    v7 = v25;
    v25 = *(_QWORD *)(v23 + 32);
  }
  if ( v7 )
    goto LABEL_18;
LABEL_34:
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !v7 )
    return 0;
  while ( 1 )
  {
LABEL_18:
    v10 = *(unsigned __int8 *)(v7 + 80);
    if ( v26 )
    {
      if ( (unsigned __int8)v10 > 0x14u )
        goto LABEL_17;
      v8 = 1182720;
      if ( !_bittest64(&v8, v10) )
        goto LABEL_17;
    }
    else if ( (_BYTE)v10 != *(_BYTE *)(a1 + 80) )
    {
      goto LABEL_17;
    }
    v9 = *(_BYTE *)(v7 + 81) & 0x10;
    if ( v24 )
    {
      if ( !v9 || v24 != *(_QWORD *)(v7 + 64) )
        goto LABEL_17;
    }
    else if ( v5 )
    {
      if ( v9 || v5 != *(_QWORD *)(v7 + 64) )
        goto LABEL_17;
    }
    else if ( v9 || *(_QWORD *)(v7 + 64) )
    {
      goto LABEL_17;
    }
    v16 = v7;
    if ( (_BYTE)v10 != 17 )
      break;
    v16 = *(_QWORD *)(v7 + 88);
    if ( v16 )
      break;
LABEL_17:
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return 0;
  }
  while ( 1 )
  {
    v18 = sub_87D1A0(v16, v28);
    if ( v18 )
    {
      if ( *(_QWORD *)(v18 + 32) == v25 && (unsigned int)sub_880F30(v16, a2) )
        return v16;
    }
    if ( (_BYTE)v10 == 17 )
    {
      v16 = *(_QWORD *)(v16 + 8);
      if ( v16 )
        continue;
    }
    goto LABEL_17;
  }
}
