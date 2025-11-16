// Function: sub_2B0C4B0
// Address: 0x2b0c4b0
//
__int64 __fastcall sub_2B0C4B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v6; // r10
  __int64 i; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r9

  v4 = a2;
  v6 = (a3 - 1) / 2;
  if ( a2 < v6 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * (i + 1);
      v10 = 16 * (i + 1);
      v9 = *(_QWORD *)(a1 + v10);
      v11 = *(_QWORD *)(a1 + v10 - 8);
      v12 = *(_QWORD *)(v9 + 184);
      if ( v12 )
      {
        v13 = *(_QWORD *)(v11 + 184);
        if ( v13 )
          break;
      }
      if ( *(_DWORD *)(v9 + 200) < *(_DWORD *)(v11 + 200) )
        goto LABEL_4;
      *(_QWORD *)(a1 + 8 * i) = v9;
      if ( a2 >= v6 )
        goto LABEL_11;
LABEL_6:
      ;
    }
    if ( *(_DWORD *)(v12 + 200) < *(_DWORD *)(v13 + 200) )
    {
LABEL_4:
      --a2;
      v9 = *(_QWORD *)(a1 + 8 * a2);
    }
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( a2 >= v6 )
      goto LABEL_11;
    goto LABEL_6;
  }
LABEL_11:
  if ( (a3 & 1) != 0 || (a3 - 2) / 2 != a2 )
    return sub_2B0C420(a1, a2, v4, a4);
  *(_QWORD *)(a1 + 8 * a2) = *(_QWORD *)(a1 + 8 * (2 * a2 + 2) - 8);
  return sub_2B0C420(a1, 2 * a2 + 1, v4, a4);
}
