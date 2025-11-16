// Function: sub_860500
// Address: 0x860500
//
void __fastcall sub_860500(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 i; // rbx
  char v7; // al
  __int64 v8; // rdi
  __int64 v9; // rdx
  _BOOL4 v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r15
  char v14; // cl
  __int64 v15; // rdi
  FILE *v16; // rsi
  char v17; // al
  char v18; // si
  unsigned __int8 v19; // al
  int v20; // [rsp+Ch] [rbp-34h]

  for ( i = *(_QWORD *)(a1 + 104); i; i = *(_QWORD *)(i + 112) )
  {
    while ( 1 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
      {
        v7 = *(_BYTE *)(i + 177);
        if ( (v7 & 0x20) == 0 )
        {
          v8 = *(_QWORD *)(*(_QWORD *)(i + 168) + 152LL);
          if ( v8 )
          {
            if ( (*(_BYTE *)(v8 + 29) & 0x20) == 0 )
              break;
          }
        }
      }
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_7;
    }
    v9 = a3;
    v10 = (v7 & 4) != 0;
    if ( !a3 )
      v9 = v10;
    sub_860500(v8, a2, v9);
  }
LABEL_7:
  if ( *(_BYTE *)(a1 + 28) == 6 )
  {
    v11 = *(_QWORD *)(a1 + 32);
    if ( (*(_BYTE *)(v11 + 89) & 1) == a2 )
    {
      v12 = *(_QWORD *)(a1 + 144);
      v20 = *(_DWORD *)(v11 + 176) & 0x20A000;
      while ( v12 )
      {
        v13 = *(_QWORD *)v12;
        if ( sub_860410(v12) || v13 && (*(_BYTE *)(v13 + 81) & 2) != 0 )
          goto LABEL_30;
        if ( (*(_BYTE *)(v12 + 193) & 0x10) != 0 )
          goto LABEL_30;
        v14 = *(_BYTE *)(v12 + 192);
        if ( (*(_BYTE *)(v12 + 88) & 4) == 0 && (*(_BYTE *)(v12 + 192) & 0x8A) != 0x82 )
          goto LABEL_30;
        if ( !v13 )
          goto LABEL_30;
        v15 = *(_QWORD *)(v13 + 96);
        if ( v15 )
        {
          if ( (*(_BYTE *)(v15 + 80) & 0xA) != 0 )
            goto LABEL_30;
        }
        if ( a3 && (v14 & 2) != 0 )
          goto LABEL_30;
        if ( a2 )
        {
          if ( (v14 & 0xA) != 2 )
          {
            v16 = (FILE *)(v13 + 48);
            if ( dword_4D04964 )
              sub_6854C0(0x149u, v16, v13);
            else
              sub_685490(0x149u, v16, v13);
          }
          goto LABEL_30;
        }
        if ( (*(_BYTE *)(v12 + 88) & 4) == 0 )
          goto LABEL_33;
        v18 = *(_BYTE *)(v12 + 192) & 0x8A;
        if ( !(v20 | (unsigned int)sub_736990(v12)) )
        {
          if ( *(char *)(v12 + 192) < 0 )
          {
            if ( dword_4F077BC )
              v19 = 2 * (*(_BYTE *)(v12 + 172) != 1) + 5;
            else
              v19 = 7;
            goto LABEL_46;
          }
          if ( *(_BYTE *)(v12 + 172) != 1 )
          {
            v19 = HIDWORD(qword_4F077B4) == 0 ? 7 : 5;
LABEL_46:
            sub_6853B0(v19, 0x72u, (FILE *)(v13 + 48), v13);
            goto LABEL_30;
          }
        }
        if ( v18 == -126 )
        {
LABEL_33:
          if ( dword_4D04964 )
          {
            sub_6853B0(unk_4F07471, 0x339u, (FILE *)(v13 + 48), v13);
          }
          else if ( unk_4D03FE8 && sub_7E4A00(*(_QWORD *)(a1 + 32)) )
          {
            sub_685490(0x339u, (FILE *)(v13 + 48), v13);
          }
          v17 = *(_BYTE *)(v12 + 88);
          *(_BYTE *)(v12 + 172) = 1;
          *(_BYTE *)(v12 + 88) = v17 & 0x8F | 0x20;
        }
LABEL_30:
        v12 = *(_QWORD *)(v12 + 112);
      }
    }
  }
}
