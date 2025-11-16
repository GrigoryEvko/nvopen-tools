// Function: sub_772A90
// Address: 0x772a90
//
__int64 __fastcall sub_772A90(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v6; // rax
  char v7; // dl
  __int64 v9; // r12
  char v10; // al
  __int64 v11; // rcx
  unsigned __int8 v12; // dl
  __int64 v13; // rdx
  int v14; // [rsp+8h] [rbp-18h] BYREF
  _DWORD v15[5]; // [rsp+Ch] [rbp-14h] BYREF

  v6 = *a4;
  v7 = **a4;
  if ( v7 != 13 )
    goto LABEL_2;
  v11 = *((_QWORD *)v6 + 1);
  v12 = *(_BYTE *)(v11 + 24);
  if ( v12 == 4 )
  {
    *v6 = 8;
    v13 = *(_QWORD *)(v11 + 56);
    *((_QWORD *)v6 + 1) = v13;
    goto LABEL_18;
  }
  if ( v12 <= 4u )
  {
    if ( v12 == 2 )
    {
      *v6 = 2;
      v13 = *(_QWORD *)(v11 + 56);
      *((_QWORD *)v6 + 1) = v13;
      goto LABEL_18;
    }
    if ( v12 == 3 )
    {
      *v6 = 7;
      v13 = *(_QWORD *)(v11 + 56);
      *((_QWORD *)v6 + 1) = v13;
LABEL_18:
      if ( (*(_BYTE *)(v13 - 8) & 1) == 0 )
        goto LABEL_3;
      *((_DWORD *)v6 + 4) = 0;
      v7 = *v6;
LABEL_2:
      if ( v7 != 11 )
        goto LABEL_3;
      goto LABEL_7;
    }
LABEL_23:
    if ( (*(_BYTE *)(v11 - 8) & 1) != 0 )
      *((_DWORD *)v6 + 4) = 0;
    goto LABEL_3;
  }
  if ( v12 != 20 )
    goto LABEL_23;
  *v6 = 11;
  v9 = *(_QWORD *)(v11 + 56);
  *((_QWORD *)v6 + 1) = v9;
  if ( (*(_BYTE *)(v9 - 8) & 1) == 0 )
    goto LABEL_8;
  *((_DWORD *)v6 + 4) = 0;
LABEL_7:
  v9 = *((_QWORD *)v6 + 1);
LABEL_8:
  v10 = *(_BYTE *)(v9 + 174);
  if ( v10 == 1 )
  {
    if ( (unsigned int)sub_72F310(v9, 1)
      || (unsigned int)sub_72F500(v9, *(_QWORD *)(*(_QWORD *)(v9 + 40) + 32LL), (char *)v15, 1, 1) )
    {
      goto LABEL_3;
    }
    v10 = *(_BYTE *)(v9 + 174);
  }
  if ( v10 != 2 && (v10 != 5 || !(unsigned int)sub_72F790(v9, v15, &v14)) )
  {
LABEL_3:
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
    return 1;
  }
  *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  return 1;
}
