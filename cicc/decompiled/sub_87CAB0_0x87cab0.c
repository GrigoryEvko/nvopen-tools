// Function: sub_87CAB0
// Address: 0x87cab0
//
__int64 __fastcall sub_87CAB0(
        __int64 a1,
        FILE *a2,
        __int64 a3,
        int a4,
        unsigned int a5,
        unsigned int a6,
        int a7,
        _DWORD *a8,
        int *a9)
{
  __int64 v10; // r12
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // r12
  __int64 v16; // rax
  int v17; // eax
  unsigned int v18; // edi
  FILE *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // eax
  int v23; // [rsp+0h] [rbp-50h]
  int v25; // [rsp+10h] [rbp-40h] BYREF
  int v26; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v27[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = a1;
  v27[0] = 0;
  if ( a8 )
  {
    *a8 = 0;
    if ( *(_BYTE *)(a1 + 140) != 12 )
      goto LABEL_5;
  }
  else if ( *(_BYTE *)(a1 + 140) != 12 )
  {
    goto LABEL_11;
  }
  do
    v10 = *(_QWORD *)(v10 + 160);
  while ( *(_BYTE *)(v10 + 140) == 12 );
  if ( a8 )
  {
LABEL_5:
    v12 = sub_697930(v10, 1u, a4, a7, (int)a2, &v25, 0, &v26);
    if ( v25 )
    {
LABEL_6:
      *a8 = 1;
      v13 = 1;
      v14 = 0;
      goto LABEL_7;
    }
    goto LABEL_12;
  }
LABEL_11:
  v12 = sub_697930(v10, 1u, a4, a7, (int)a2, &v25, (__int64)v27, &v26);
  if ( v25 )
  {
    v20 = v10;
    v14 = 0;
    sub_685360(0x153u, a2, v20);
    v13 = 1;
    goto LABEL_7;
  }
LABEL_12:
  if ( !v12 )
  {
    if ( v26 )
    {
      v14 = 0;
      v13 = 0;
      goto LABEL_7;
    }
    if ( a8 )
      goto LABEL_6;
LABEL_23:
    v18 = 291;
LABEL_24:
    v19 = (FILE *)sub_67DA80(v18, a2, v10);
    sub_87CA90(v27[0], v19);
    v14 = 0;
    sub_685910((__int64)v19, v19);
    v13 = 1;
    goto LABEL_7;
  }
  if ( (*(_BYTE *)(v12 + 104) & 1) != 0 )
  {
    v17 = sub_8796F0(v12);
  }
  else
  {
    v16 = *(_QWORD *)(v12 + 88);
    if ( *(_BYTE *)(v12 + 80) == 20 )
      v16 = *(_QWORD *)(v16 + 176);
    v17 = (*(_BYTE *)(v16 + 208) & 4) != 0;
  }
  if ( v17 )
  {
    if ( a8 )
      goto LABEL_6;
    if ( (*(_BYTE *)(v12 + 104) & 1) != 0 )
    {
      v22 = sub_8796F0(v12);
    }
    else
    {
      v21 = *(_QWORD *)(v12 + 88);
      if ( *(_BYTE *)(v12 + 80) == 20 )
        v21 = *(_QWORD *)(v21 + 176);
      v22 = (*(_BYTE *)(v21 + 208) & 4) != 0;
    }
    v18 = 3126;
    if ( v22 )
      goto LABEL_24;
    goto LABEL_23;
  }
  v14 = *(_QWORD *)(v12 + 88);
  if ( (*(_BYTE *)(v14 + 206) & 0x10) != 0 )
  {
    v17 = 1;
  }
  else if ( (*(_BYTE *)(v14 + 194) & 2) != 0 )
  {
    a5 = 0;
    v14 = 0;
  }
  v23 = v17;
  sub_8769C0(v12, a2, a3, 0, a5, 1, a6, 0, a8);
  v13 = v23;
LABEL_7:
  if ( a9 )
    *a9 = v13;
  return v14;
}
