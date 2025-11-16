// Function: sub_22C0B50
// Address: 0x22c0b50
//
__int64 __fastcall sub_22C0B50(__int64 a1, char *a2, unsigned int a3, char a4)
{
  char v5; // al
  unsigned int v6; // esi
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax

  v5 = *a2;
  if ( *a2 == 4 )
    goto LABEL_8;
  v6 = a3;
  if ( v5 != 5 )
    goto LABEL_3;
  if ( a4 || (v6 = a3, sub_9876C0((__int64 *)a2 + 1)) )
  {
LABEL_8:
    v8 = *((_DWORD *)a2 + 4);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 > 0x40 )
    {
      sub_C43780(a1, (const void **)a2 + 1);
      v10 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 24) = v10;
      if ( v10 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)a1 = *((_QWORD *)a2 + 1);
      v9 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 24) = v9;
      if ( v9 <= 0x40 )
      {
LABEL_10:
        *(_QWORD *)(a1 + 16) = *((_QWORD *)a2 + 3);
        return a1;
      }
    }
    sub_C43780(a1 + 16, (const void **)a2 + 3);
    return a1;
  }
  v5 = *a2;
LABEL_3:
  if ( v5 == 2 )
  {
    sub_AD8380(a1, *((_QWORD *)a2 + 1));
    return a1;
  }
  else
  {
    if ( v5 )
      sub_AADB10(a1, v6, 1);
    else
      sub_AADB10(a1, v6, 0);
    return a1;
  }
}
