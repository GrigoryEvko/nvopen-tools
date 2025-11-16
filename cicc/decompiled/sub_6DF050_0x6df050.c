// Function: sub_6DF050
// Address: 0x6df050
//
__int64 __fastcall sub_6DF050(__int64 *a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  unsigned __int8 v3; // r13
  char v5; // cl
  __int64 v6; // rax
  char v7; // dl
  unsigned int v8; // r15d
  unsigned __int8 v9; // r14
  unsigned int v10; // r8d
  unsigned int v11; // eax
  unsigned __int8 v12; // al
  unsigned __int64 v13; // rsi
  int v14; // edi
  int v15; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = *a1;
  v2 = *(_QWORD *)(*(_QWORD *)(a1[9] + 16) + 56LL);
  v3 = *(_BYTE *)(v2 + 137);
  if ( (*((_BYTE *)a1 + 25) & 3) == 0 )
    v1 = sub_73D720(*a1);
  while ( *(_BYTE *)(v1 + 140) == 12 )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (unsigned int)sub_8D3D40(v1) || (unsigned int)sub_8D28B0(v1) )
    return v1;
  v5 = *(_BYTE *)(v1 + 140);
  if ( v5 == 12 )
  {
    v6 = v1;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v7 = *(_BYTE *)(v6 + 140);
    }
    while ( v7 == 12 );
  }
  else
  {
    v7 = *(_BYTE *)(v1 + 140);
  }
  if ( !v7 )
    return v1;
  v8 = v3;
  v9 = *(_BYTE *)(v1 + 160);
  if ( dword_4F077C4 == 2 )
  {
    sub_622920(v9, v16, &v15);
    v5 = *(_BYTE *)(v1 + 140);
    if ( v16[0] * (unsigned __int64)dword_4F06BA0 < v3 )
      v8 = LODWORD(v16[0]) * dword_4F06BA0;
  }
  if ( v5 == 2 && (*(_BYTE *)(v1 + 161) & 8) != 0 && v9 > 4u )
    goto LABEL_39;
  if ( !dword_4F077C0 )
  {
    if ( !dword_4F077BC )
      goto LABEL_41;
    if ( (_DWORD)qword_4F077B4 )
    {
LABEL_17:
      if ( (*(_BYTE *)(v2 + 144) & 8) == 0 )
        goto LABEL_18;
LABEL_42:
      if ( unk_4F06B20 * dword_4F06BA0 < v8 )
        goto LABEL_37;
      goto LABEL_43;
    }
    v13 = qword_4F077A8;
LABEL_28:
    if ( v9 > 4u && v13 <= 0x9D07 )
      goto LABEL_37;
    v10 = dword_4F06BA0;
    v14 = unk_4F06B20;
    if ( (*(_BYTE *)(v2 + 144) & 8) != 0 )
      goto LABEL_42;
    goto LABEL_31;
  }
  if ( !(_DWORD)qword_4F077B4 )
  {
    v13 = qword_4F077A8;
    if ( qword_4F077A8 <= 0x9C3Fu && *(unsigned __int8 *)(v2 + 137) == unk_4F06B10 * dword_4F06BA0 )
      goto LABEL_37;
    if ( !dword_4F077BC )
    {
      if ( (*(_BYTE *)(v2 + 144) & 8) != 0 )
        goto LABEL_42;
      v10 = dword_4F06BA0;
      v14 = unk_4F06B20;
LABEL_31:
      v11 = v10 * v14;
      goto LABEL_32;
    }
    goto LABEL_28;
  }
  if ( dword_4F077BC )
    goto LABEL_17;
LABEL_41:
  if ( (*(_BYTE *)(v2 + 144) & 8) != 0 )
    goto LABEL_42;
  v10 = dword_4F06BA0;
  v14 = unk_4F06B20;
  if ( !(_DWORD)qword_4F077B4 )
    goto LABEL_31;
LABEL_18:
  v10 = dword_4F06BA0;
  v11 = dword_4F06BA0 * unk_4F06B20;
  if ( *(_QWORD *)(v2 + 176) == dword_4F06BA0 * unk_4F06B20 )
  {
LABEL_19:
    v12 = 6;
    goto LABEL_20;
  }
LABEL_32:
  if ( v8 >= v11 )
  {
    if ( v8 == v11 )
      goto LABEL_19;
    if ( dword_4F077C0 && qword_4F077A8 == 40000 && v9 == 10 && unk_4F06B00 * v10 > v8 )
    {
      v12 = 9;
      return sub_72BA30(v12);
    }
LABEL_37:
    if ( v5 == 2 && (*(_BYTE *)(v1 + 161) & 8) != 0 )
    {
LABEL_39:
      v12 = v9;
      return sub_72BA30(v12);
    }
    return v1;
  }
LABEL_43:
  v12 = 5;
LABEL_20:
  if ( v12 == v9 )
    goto LABEL_37;
  return sub_72BA30(v12);
}
