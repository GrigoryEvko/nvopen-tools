// Function: sub_2FEF140
// Address: 0x2fef140
//
__int64 __fastcall sub_2FEF140(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = a1 + 16;
  if ( qword_5027E30 )
  {
    *(_QWORD *)a1 = v2;
    sub_2FEEBD0((__int64 *)a1, (_BYTE *)qword_5027E28, qword_5027E28 + qword_5027E30);
    return a1;
  }
  else
  {
    if ( *(_BYTE *)(a2 + 848) && *(_DWORD *)(a2 + 824) == 3 )
    {
      *(_QWORD *)a1 = v2;
      sub_2FEEBD0((__int64 *)a1, *(_BYTE **)(a2 + 696), *(_QWORD *)(a2 + 696) + *(_QWORD *)(a2 + 704));
    }
    else
    {
      *(_QWORD *)a1 = v2;
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
    }
    return a1;
  }
}
