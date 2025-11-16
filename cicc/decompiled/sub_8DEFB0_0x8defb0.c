// Function: sub_8DEFB0
// Address: 0x8defb0
//
_BOOL8 __fastcall sub_8DEFB0(__int64 a1, __int64 a2, _BOOL4 a3, int *a4)
{
  char v7; // al
  char v8; // dl
  int v9; // r13d
  int v10; // eax
  int v12; // eax
  int v13; // ebx
  __int64 v14; // rcx
  __int64 v15; // r8
  char v16; // al
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  int v22; // [rsp+8h] [rbp-38h]
  int v23; // [rsp+Ch] [rbp-34h]

  v22 = 0;
  v23 = 0;
  while ( 1 )
  {
    v7 = *(_BYTE *)(a2 + 140);
    if ( (v7 & 0xFB) != 8 )
    {
      v8 = *(_BYTE *)(a1 + 140);
      if ( (v8 & 0xFB) != 8 )
        goto LABEL_14;
      v9 = 0;
LABEL_5:
      v10 = sub_8D4C10(a1, dword_4F077C4 != 2);
      if ( !a3 && (v10 & ~v9) != 0 )
        goto LABEL_7;
      v8 = *(_BYTE *)(a1 + 140);
      v9 &= ~v10;
      goto LABEL_11;
    }
    v12 = sub_8D4C10(a2, dword_4F077C4 != 2);
    v8 = *(_BYTE *)(a1 + 140);
    v9 = v12;
    if ( (v8 & 0xFB) == 8 )
      goto LABEL_5;
LABEL_11:
    v13 = 1;
    if ( !v9 )
      v13 = v23;
    v7 = *(_BYTE *)(a2 + 140);
    v23 = v13;
LABEL_14:
    if ( v7 == 12 )
    {
      do
        a2 = *(_QWORD *)(a2 + 160);
      while ( *(_BYTE *)(a2 + 140) == 12 );
    }
    if ( v8 == 12 )
    {
      do
        a1 = *(_QWORD *)(a1 + 160);
      while ( *(_BYTE *)(a1 + 140) == 12 );
    }
    if ( (unsigned int)sub_8D2F30(a2, a1) )
    {
      if ( *(_QWORD *)(a2 + 128) != *(_QWORD *)(a1 + 128) )
        goto LABEL_20;
      a2 = sub_8D46C0(a2);
      a1 = sub_8D46C0(a1);
      goto LABEL_31;
    }
    v16 = *(_BYTE *)(a2 + 140);
    if ( v16 != 13 )
      break;
    if ( *(_BYTE *)(a1 + 140) != 13 )
      goto LABEL_33;
    v17 = sub_8D4890(a2);
    v18 = sub_8D4890(a1);
    if ( !(unsigned int)sub_8DED30(v18, v17, 17, v19, v20) )
    {
LABEL_20:
      a3 = 0;
      goto LABEL_7;
    }
    a2 = sub_8D4870(a2);
    a1 = sub_8D4870(a1);
LABEL_31:
    v22 = (int)&dword_400000;
  }
  if ( v16 == 8 && *(_BYTE *)(a1 + 140) == 8 )
  {
    if ( ((*(_WORD *)(a2 + 168) & 0x180) != 0
       || (*(_WORD *)(a1 + 168) & 0x180) != 0
       || *(_QWORD *)(a2 + 176) != *(_QWORD *)(a1 + 176))
      && (!dword_4D047EC || (*(_BYTE *)(a2 + 169) & 2) == 0 && (*(_BYTE *)(a1 + 169) & 2) == 0) )
    {
      goto LABEL_20;
    }
    a2 = sub_8D4050(a2);
    a1 = sub_8D4050(a1);
    goto LABEL_31;
  }
LABEL_33:
  a3 = 1;
  if ( a2 != a1 )
    a3 = sub_8DED30(a1, a2, v22 | 0x13u, v14, v15) != 0;
LABEL_7:
  if ( a4 )
    *a4 = v23;
  return a3;
}
