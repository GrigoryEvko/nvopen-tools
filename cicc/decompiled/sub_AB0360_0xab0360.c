// Function: sub_AB0360
// Address: 0xab0360
//
__int64 __fastcall sub_AB0360(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v5; // eax
  unsigned int v6; // eax
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax

  if ( a4 == 1 )
  {
    if ( !sub_AAFBB0(a2) && sub_AAFBB0(a3) )
      goto LABEL_4;
    if ( sub_AAFBB0(a2) && !sub_AAFBB0(a3) )
    {
LABEL_23:
      sub_AAF450(a1, a3);
      return a1;
    }
  }
  else if ( a4 == 2 )
  {
    if ( !sub_AB0120(a2) && sub_AB0120(a3) )
    {
      sub_AAF450(a1, a2);
      return a1;
    }
    if ( sub_AB0120(a2) && !sub_AB0120(a3) )
      goto LABEL_23;
  }
  if ( !sub_AB01D0(a2, a3) )
  {
    v8 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 > 0x40 )
      sub_C43780(a1, a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v9 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v9;
    if ( v9 > 0x40 )
      sub_C43780(a1 + 16, a3 + 16);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
    return a1;
  }
LABEL_4:
  v5 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v5;
  if ( v5 > 0x40 )
  {
    sub_C43780(a1, a2);
    v10 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v10;
    if ( v10 <= 0x40 )
      goto LABEL_6;
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v6 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v6;
    if ( v6 <= 0x40 )
    {
LABEL_6:
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      return a1;
    }
  }
  sub_C43780(a1 + 16, a2 + 16);
  return a1;
}
