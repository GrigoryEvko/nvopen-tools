// Function: sub_38D3EB0
// Address: 0x38d3eb0
//
__int64 __fastcall sub_38D3EB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax

  v4 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 )
    goto LABEL_8;
  if ( (*(_BYTE *)(a2 + 9) & 0xC) != 8 )
    goto LABEL_3;
  *(_BYTE *)(a2 + 8) |= 4u;
  v6 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
  v7 = v6 | *(_QWORD *)a2 & 7LL;
  *(_QWORD *)a2 = v7;
  if ( !v6 )
    goto LABEL_3;
  v4 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 || (v4 = 0, (*(_BYTE *)(a2 + 9) & 0xC) != 8) )
  {
LABEL_8:
    v8 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 )
      goto LABEL_10;
  }
  else
  {
    *(_BYTE *)(a2 + 8) |= 4u;
    v4 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
    *(_QWORD *)a2 = v4 | *(_QWORD *)a2 & 7LL;
    v8 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 )
    {
LABEL_10:
      if ( v8 != v4 )
        goto LABEL_3;
      goto LABEL_11;
    }
  }
  v8 = 0;
  if ( (*(_BYTE *)(a3 + 9) & 0xC) != 8 )
    goto LABEL_10;
  *(_BYTE *)(a3 + 8) |= 4u;
  v10 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a3 + 24));
  *(_QWORD *)a3 = v10 | *(_QWORD *)a3 & 7LL;
  if ( v10 != v4 )
    goto LABEL_3;
LABEL_11:
  if ( (*(_BYTE *)(a2 + 9) & 0xC) != 8 && (*(_BYTE *)(a3 + 9) & 0xC) != 8 )
  {
    v9 = *(_QWORD *)(a2 + 24) - *(_QWORD *)(a3 + 24);
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v9;
    return a1;
  }
LABEL_3:
  *(_BYTE *)(a1 + 8) = 0;
  return a1;
}
