// Function: sub_774620
// Address: 0x774620
//
__int64 __fastcall sub_774620(__int64 a1, __int64 a2, __int64 a3, char **a4, _WORD *a5)
{
  char *v6; // rax
  char v7; // dl
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int8 v12; // dl
  __int64 v13; // rdx

  v6 = *a4;
  v7 = **a4;
  if ( v7 != 13 )
    goto LABEL_2;
  v11 = *((_QWORD *)v6 + 1);
  v12 = *(_BYTE *)(v11 + 24);
  if ( v12 == 4 )
  {
    *v6 = 8;
    v10 = *(_QWORD *)(v11 + 56);
    *((_QWORD *)v6 + 1) = v10;
    if ( (*(_BYTE *)(v10 - 8) & 1) == 0 )
    {
LABEL_10:
      v8 = *(_QWORD *)(v10 + 128);
      goto LABEL_5;
    }
    *((_DWORD *)v6 + 4) = 0;
LABEL_9:
    v10 = *((_QWORD *)v6 + 1);
    goto LABEL_10;
  }
  if ( v12 > 4u )
  {
    if ( v12 == 20 )
    {
      *v6 = 11;
      v13 = *(_QWORD *)(v11 + 56);
      *((_QWORD *)v6 + 1) = v13;
      goto LABEL_16;
    }
LABEL_18:
    if ( (*(_BYTE *)(v11 - 8) & 1) != 0 )
      *((_DWORD *)v6 + 4) = 0;
    goto LABEL_20;
  }
  if ( v12 == 2 )
  {
    *v6 = 2;
    v13 = *(_QWORD *)(v11 + 56);
    *((_QWORD *)v6 + 1) = v13;
    goto LABEL_16;
  }
  if ( v12 != 3 )
    goto LABEL_18;
  *v6 = 7;
  v13 = *(_QWORD *)(v11 + 56);
  *((_QWORD *)v6 + 1) = v13;
LABEL_16:
  if ( (*(_BYTE *)(v13 - 8) & 1) == 0 )
    goto LABEL_20;
  *((_DWORD *)v6 + 4) = 0;
  v7 = *v6;
LABEL_2:
  if ( v7 == 8 )
    goto LABEL_9;
  if ( v7 == 37 )
  {
    v8 = *(_QWORD *)(*((_QWORD *)v6 + 1) + 104LL);
LABEL_5:
    sub_620D80(a5, v8);
    return 1;
  }
LABEL_20:
  result = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    return 0;
  }
  return result;
}
