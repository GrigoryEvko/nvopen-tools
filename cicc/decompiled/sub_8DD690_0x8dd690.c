// Function: sub_8DD690
// Address: 0x8dd690
//
__int64 __fastcall sub_8DD690(__int64 a1, __int64 a2, int a3, __m128i *a4, __int64 a5, int *a6)
{
  __int64 v7; // r14
  char v10; // al
  __int64 result; // rax
  int v12; // edx
  int v13; // eax
  __m128i *v14; // rdx
  __int64 v15; // rax
  char v16; // al
  char v17; // cl
  __int64 v18; // rax
  int v20[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v7 = a2;
  v10 = *(_BYTE *)(a1 + 12);
  v20[0] = 0;
  if ( (v10 & 0x10) != 0 && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) || *(_QWORD *)a1 )
  {
LABEL_3:
    result = sub_8DBE70(a5);
    if ( !(_DWORD)result )
    {
      if ( !a6 )
        return result;
      v12 = v20[0];
      if ( !v20[0] )
        v12 = 458;
      goto LABEL_7;
    }
    goto LABEL_10;
  }
  if ( !word_4D04898 )
    goto LABEL_10;
  if ( !a5 )
  {
    if ( (v10 & 0x20) == 0 )
      goto LABEL_10;
    goto LABEL_3;
  }
  v13 = sub_8D2960(a5);
  v14 = a4;
  if ( v13 )
  {
    if ( (unsigned int)sub_8DD4B0(a2, a3, a4, a5, v20) )
      goto LABEL_10;
    goto LABEL_3;
  }
  if ( (*(_BYTE *)(a1 + 12) & 0x20) == 0 )
    goto LABEL_10;
  if ( dword_4F04C44 != -1
    || (v15 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v15 + 6) & 6) != 0)
    || *(_BYTE *)(v15 + 4) == 12 )
  {
    if ( (unsigned int)sub_8DBE70(a2) || (unsigned int)sub_8DBE70(a5) )
      goto LABEL_10;
    v14 = a4;
  }
  while ( 1 )
  {
    v16 = *(_BYTE *)(v7 + 140);
    if ( v16 != 12 )
      break;
    v7 = *(_QWORD *)(v7 + 160);
  }
  if ( v16 )
  {
    v17 = *(_BYTE *)(a5 + 140);
    if ( v17 == 12 )
    {
      v18 = a5;
      do
      {
        v18 = *(_QWORD *)(v18 + 160);
        v17 = *(_BYTE *)(v18 + 140);
      }
      while ( v17 == 12 );
    }
    if ( v17 && (!a3 || !sub_8D2660(v14[8].m128i_i64[0]) || !(unsigned int)sub_8D2EF0(a5) && !sub_8D3D10(a5)) )
      goto LABEL_3;
  }
LABEL_10:
  v12 = 0;
  result = 1;
  if ( a6 )
LABEL_7:
    *a6 = v12;
  return result;
}
