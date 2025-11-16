// Function: sub_70C9E0
// Address: 0x70c9e0
//
__int64 __fastcall sub_70C9E0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  char v11; // al
  __int64 i; // rbx
  __int64 j; // r13
  __int64 v15; // rdi
  char k; // dl
  __int64 v17; // rsi
  char m; // al

  v8 = *(_QWORD *)(a1 + 144);
  if ( v8 )
  {
    if ( a3 )
    {
      v9 = *(_QWORD *)(a1 + 128);
      if ( v9 == a2 || (unsigned int)sub_8D97D0(v9, a2, 0, a4, a5) )
      {
        *(_BYTE *)(a1 + 168) |= 8u;
        if ( *(_QWORD *)(a1 + 136) )
          goto LABEL_10;
LABEL_12:
        for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        for ( j = a2; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( (unsigned int)sub_8DB040(i, j) )
        {
          if ( *(_BYTE *)(i + 140) != 6 || *(_BYTE *)(j + 140) != 6 )
            goto LABEL_10;
          v15 = *(_QWORD *)(i + 160);
          for ( k = *(_BYTE *)(v15 + 140); k == 12; k = *(_BYTE *)(v15 + 140) )
            v15 = *(_QWORD *)(v15 + 160);
          v17 = *(_QWORD *)(j + 160);
          for ( m = *(_BYTE *)(v17 + 140); m == 12; m = *(_BYTE *)(v17 + 140) )
            v17 = *(_QWORD *)(v17 + 160);
          if ( m != 7 || k != 7 || (unsigned int)sub_8DBCE0() )
            goto LABEL_10;
        }
        else if ( *(_BYTE *)(j + 140) == 6 && (unsigned int)sub_8D2E30(i) && (unsigned int)sub_8D2710(j)
               || *(_BYTE *)(i + 140) == 19 && ((unsigned int)sub_8D2E30(j) || (unsigned int)sub_8D3D10(j)) )
        {
          goto LABEL_10;
        }
        *(_BYTE *)(a1 + 168) |= 0x80u;
        goto LABEL_10;
      }
      v8 = *(_QWORD *)(a1 + 144);
    }
    v10 = sub_73DBF0(5, a2, v8);
    *(_QWORD *)(a1 + 144) = v10;
    *(_BYTE *)(v10 + 27) = (2 * (a3 & 1)) | *(_BYTE *)(v10 + 27) & 0xFD;
  }
  v11 = *(_BYTE *)(a1 + 168);
  *(_BYTE *)(a1 + 168) = v11 | 8;
  if ( !a3 )
    *(_BYTE *)(a1 + 168) = v11 | 0x28;
  if ( !*(_QWORD *)(a1 + 136) )
    goto LABEL_12;
LABEL_10:
  *(_QWORD *)(a1 + 128) = a2;
  return sub_72A160(a1);
}
