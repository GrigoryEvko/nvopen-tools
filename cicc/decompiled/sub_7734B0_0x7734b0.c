// Function: sub_7734B0
// Address: 0x7734b0
//
__int64 __fastcall sub_7734B0(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char *v6; // rax
  char v7; // si
  __int64 v8; // rdi
  unsigned __int8 v9; // dl
  __int64 v10; // rax
  char v12; // dl

  v6 = *a4;
  v7 = **a4;
  v8 = *((_QWORD *)*a4 + 1);
  if ( v7 == 48 )
  {
    v12 = *(_BYTE *)(v8 + 8);
    if ( v12 == 1 )
    {
      *v6 = 2;
      v8 = *(_QWORD *)(v8 + 32);
      v7 = 2;
      *((_QWORD *)v6 + 1) = v8;
    }
    else if ( v12 == 2 )
    {
      *v6 = 59;
      v8 = *(_QWORD *)(v8 + 32);
      v7 = 59;
      *((_QWORD *)v6 + 1) = v8;
    }
    else
    {
      if ( v12 )
        sub_721090();
      *v6 = 6;
      v8 = *(_QWORD *)(v8 + 32);
      v7 = 6;
      *((_QWORD *)v6 + 1) = v8;
    }
  }
  else if ( v7 == 13 )
  {
    v9 = *(_BYTE *)(v8 + 24);
    if ( v9 == 4 )
    {
      *v6 = 8;
      v8 = *(_QWORD *)(v8 + 56);
      v7 = 8;
      *((_QWORD *)v6 + 1) = v8;
    }
    else if ( v9 > 4u )
    {
      if ( v9 == 20 )
      {
        *v6 = 11;
        v8 = *(_QWORD *)(v8 + 56);
        v7 = 11;
        *((_QWORD *)v6 + 1) = v8;
      }
    }
    else if ( v9 == 2 )
    {
      *v6 = 2;
      v8 = *(_QWORD *)(v8 + 56);
      v7 = 2;
      *((_QWORD *)v6 + 1) = v8;
    }
    else if ( v9 == 3 )
    {
      *v6 = 7;
      v8 = *(_QWORD *)(v8 + 56);
      v7 = 7;
      *((_QWORD *)v6 + 1) = v8;
    }
    if ( (*(_BYTE *)(v8 - 8) & 1) != 0 )
      *((_DWORD *)v6 + 4) = 0;
  }
  v10 = sub_72A270(v8, v7);
  if ( v10 && (*(_BYTE *)(v10 + 88) & 0x70) == 0x10 )
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    return 1;
  }
  else
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
    return 1;
  }
}
