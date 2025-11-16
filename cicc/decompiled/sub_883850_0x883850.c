// Function: sub_883850
// Address: 0x883850
//
__int64 __fastcall sub_883850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 i; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 result; // rax
  unsigned __int8 v16; // al
  unsigned __int8 v17; // al
  __int64 v18; // [rsp+8h] [rbp-38h]

  if ( !a3 )
    return sub_87D550(a1);
  v9 = **(_QWORD **)(a5 + 88);
  v10 = *(_QWORD *)(a5 + 64);
  if ( v10 == a2 || v10 && a2 && dword_4F07588 && (v11 = *(_QWORD *)(v10 + 32), *(_QWORD *)(a2 + 32) == v11) && v11 )
  {
LABEL_22:
    result = *(_BYTE *)(a5 + 96) & 3;
    if ( *(_BYTE *)(v9 + 80) != 17 || (*(_BYTE *)(a5 + 96) & 4) != 0 )
      return result;
    goto LABEL_25;
  }
  if ( (*(_BYTE *)(a5 + 96) & 8) != 0 )
  {
    for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v18 = **(_QWORD **)(a5 + 88);
    a5 = sub_883800(*(_QWORD *)(*(_QWORD *)i + 96LL) + 192LL, *(_QWORD *)a5);
    if ( a5 )
    {
      v9 = v18;
      do
      {
        v14 = *(_QWORD *)(a5 + 64);
        if ( v14 == a2
          || v14 && a2 && dword_4F07588 && (v13 = *(_QWORD *)(v14 + 32), *(_QWORD *)(a2 + 32) == v13) && v13 )
        {
          if ( *(_BYTE *)(a5 + 80) == 16 && **(_QWORD **)(a5 + 88) == v18 )
            goto LABEL_22;
        }
        a5 = *(_QWORD *)(a5 + 8);
      }
      while ( a5 );
    }
  }
LABEL_25:
  v16 = sub_87D550(a1);
  if ( a6 )
  {
    v17 = sub_87D7F0(v16, *(__int64 ****)a6);
    v16 = sub_87D630(v17, **(_QWORD **)(a6 + 8), *(_QWORD *)(a6 + 16));
  }
  return sub_87D630(v16, a3, a4);
}
