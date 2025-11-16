// Function: sub_7A5370
// Address: 0x7a5370
//
__int64 __fastcall sub_7A5370(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r15
  __int64 v11; // rdi
  char v12; // al
  char v13; // si
  __int64 v14; // rdi
  unsigned int v15; // r13d
  __int64 v16; // rsi
  char v18; // al
  __int64 i; // r15
  __int64 v20; // rdx
  const __m128i **v21; // r10
  __int64 j; // r15
  __int64 v23; // rax
  __int64 k; // rsi
  unsigned int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-60h]
  const __m128i **v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h] BYREF
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h]

  v28 = 0;
  v10 = *a4;
  v30 = 0;
  v29 = 0;
  v28 = sub_823970(0);
  v11 = v28;
  v12 = *(_BYTE *)(a1 + 132);
  if ( (v12 & 1) != 0 && dword_4D04880 )
  {
    v13 = *(_BYTE *)v10;
    v14 = *(_QWORD *)(v10 + 8);
    if ( *(_BYTE *)v10 == 48 )
    {
      v18 = *(_BYTE *)(v14 + 8);
      if ( v18 == 1 )
      {
        *(_BYTE *)v10 = 2;
        v14 = *(_QWORD *)(v14 + 32);
        v13 = 2;
        *(_QWORD *)(v10 + 8) = v14;
      }
      else if ( v18 == 2 )
      {
        *(_BYTE *)v10 = 59;
        v14 = *(_QWORD *)(v14 + 32);
        v13 = 59;
        *(_QWORD *)(v10 + 8) = v14;
      }
      else
      {
        if ( v18 )
          sub_721090();
        *(_BYTE *)v10 = 6;
        v14 = *(_QWORD *)(v14 + 32);
        v13 = 6;
        *(_QWORD *)(v10 + 8) = v14;
      }
    }
    if ( !sub_72A270(v14, v13) || *(_BYTE *)v10 != 6 )
      goto LABEL_6;
    for ( i = *(_QWORD *)(v10 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
      sub_8AE000(i);
    if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u && (*(_BYTE *)(i + 141) & 0x20) == 0 )
    {
      v20 = *(_QWORD *)(i + 160);
      v21 = (const __m128i **)&v28;
      for ( j = v30; v20; v30 = j )
      {
        if ( v29 == j )
        {
          v26 = v20;
          v27 = v21;
          sub_7A3E20(v21);
          v20 = v26;
          v21 = v27;
        }
        v23 = v28 + 24 * j;
        if ( v23 )
        {
          *(_BYTE *)v23 = 8;
          *(_QWORD *)(v23 + 8) = v20;
          *(_DWORD *)(v23 + 16) = 0;
        }
        v20 = *(_QWORD *)(v20 + 112);
        ++j;
      }
      for ( k = *(_QWORD *)a3; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      v25 = sub_77AFD0(a1, k, v21, (FILE *)(a3 + 28), a5, a6);
      v11 = v28;
      v15 = v25;
      v16 = 24 * v29;
    }
    else
    {
LABEL_6:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      v11 = v28;
      v15 = 0;
      v16 = 24 * v29;
    }
  }
  else
  {
    v16 = 0;
    v15 = 0;
    if ( (v12 & 0x20) == 0 )
    {
      sub_6855B0(0xAA1u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v11 = v28;
      v16 = 24 * v29;
    }
  }
  sub_823A00(v11, v16);
  return v15;
}
