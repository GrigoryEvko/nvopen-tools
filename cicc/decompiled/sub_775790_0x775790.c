// Function: sub_775790
// Address: 0x775790
//
__int64 __fastcall sub_775790(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v8; // rax
  char v9; // si
  __int64 v10; // rdi
  __int64 result; // rax
  char v12; // al
  unsigned __int8 v13; // dl
  __int64 v14; // rax

  v8 = *a4;
  v9 = **a4;
  v10 = *((_QWORD *)*a4 + 1);
  switch ( v9 )
  {
    case 13:
      v13 = *(_BYTE *)(v10 + 24);
      if ( v13 == 4 )
      {
        *v8 = 8;
        v10 = *(_QWORD *)(v10 + 56);
        v9 = 8;
        *((_QWORD *)v8 + 1) = v10;
      }
      else if ( v13 > 4u )
      {
        if ( v13 == 20 )
        {
          *v8 = 11;
          v10 = *(_QWORD *)(v10 + 56);
          v9 = 11;
          *((_QWORD *)v8 + 1) = v10;
        }
      }
      else if ( v13 == 2 )
      {
        *v8 = 2;
        v10 = *(_QWORD *)(v10 + 56);
        v9 = 2;
        *((_QWORD *)v8 + 1) = v10;
      }
      else if ( v13 == 3 )
      {
        *v8 = 7;
        v10 = *(_QWORD *)(v10 + 56);
        v9 = 7;
        *((_QWORD *)v8 + 1) = v10;
      }
      if ( (*(_BYTE *)(v10 - 8) & 1) != 0 )
        *((_DWORD *)v8 + 4) = 0;
      break;
    case 37:
      if ( (unsigned int)sub_87DF20(v10) )
      {
LABEL_4:
        *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
        return 1;
      }
      goto LABEL_21;
    case 48:
      v12 = *(_BYTE *)(v10 + 8);
      if ( v12 == 1 )
      {
        v10 = *(_QWORD *)(v10 + 32);
        v9 = 2;
      }
      else if ( v12 == 2 )
      {
        v10 = *(_QWORD *)(v10 + 32);
        v9 = 59;
      }
      else
      {
        if ( v12 )
          sub_721090();
        v10 = *(_QWORD *)(v10 + 32);
        v9 = 6;
      }
      break;
  }
  v14 = sub_72A270(v10, v9);
  if ( !v14 || (*(_BYTE *)(v14 + 89) & 4) == 0 )
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xD30u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      result = 0;
    }
    goto LABEL_24;
  }
  if ( !*(_QWORD *)v14 )
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      result = 0;
    }
    goto LABEL_24;
  }
  if ( (unsigned int)sub_884000(*(_QWORD *)v14, 1) )
    goto LABEL_4;
LABEL_21:
  result = 1;
LABEL_24:
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return result;
}
