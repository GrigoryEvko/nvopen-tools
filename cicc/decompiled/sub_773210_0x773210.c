// Function: sub_773210
// Address: 0x773210
//
__int64 __fastcall sub_773210(__int64 a1, __int64 a2, __int64 a3, char **a4, _WORD *a5)
{
  char *v6; // rax
  char v7; // dl
  __int64 v8; // rdx
  __int64 v10; // rcx
  unsigned __int8 v11; // dl
  __int64 v12; // rdx

  v6 = *a4;
  v7 = **a4;
  if ( v7 != 13 )
    goto LABEL_2;
  v10 = *((_QWORD *)v6 + 1);
  v11 = *(_BYTE *)(v10 + 24);
  if ( v11 != 4 )
  {
    if ( v11 > 4u )
    {
      if ( v11 == 20 )
      {
        *v6 = 11;
        v12 = *(_QWORD *)(v10 + 56);
        *((_QWORD *)v6 + 1) = v12;
        goto LABEL_13;
      }
    }
    else
    {
      if ( v11 == 2 )
      {
        *v6 = 2;
        v12 = *(_QWORD *)(v10 + 56);
        *((_QWORD *)v6 + 1) = v12;
        goto LABEL_13;
      }
      if ( v11 == 3 )
      {
        *v6 = 7;
        v12 = *(_QWORD *)(v10 + 56);
        *((_QWORD *)v6 + 1) = v12;
LABEL_13:
        if ( (*(_BYTE *)(v12 - 8) & 1) == 0 )
          goto LABEL_5;
        *((_DWORD *)v6 + 4) = 0;
        v7 = *v6;
LABEL_2:
        if ( v7 != 8 )
          goto LABEL_5;
        goto LABEL_3;
      }
    }
    if ( (*(_BYTE *)(v10 - 8) & 1) != 0 )
      *((_DWORD *)v6 + 4) = 0;
    goto LABEL_5;
  }
  *v6 = 8;
  v8 = *(_QWORD *)(v10 + 56);
  *((_QWORD *)v6 + 1) = v8;
  if ( (*(_BYTE *)(v8 - 8) & 1) != 0 )
  {
    *((_DWORD *)v6 + 4) = 0;
LABEL_3:
    v8 = *((_QWORD *)v6 + 1);
  }
  if ( (*(_BYTE *)(v8 + 144) & 4) != 0 )
  {
    sub_620D80(a5, *(unsigned __int8 *)(v8 + 136));
    return 1;
  }
LABEL_5:
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
  }
  return 0;
}
