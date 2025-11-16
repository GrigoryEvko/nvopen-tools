// Function: sub_2A62360
// Address: 0x2a62360
//
__int64 __fastcall sub_2A62360(__int64 a1, char *a2, __int64 a3, char a4)
{
  char v5; // r13
  unsigned int v7; // eax
  unsigned int v8; // esi
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax

  v5 = *a2;
  if ( *a2 == 4 )
    goto LABEL_8;
  v7 = sub_BCB060(a3);
  v8 = v7;
  if ( v5 != 5 )
    goto LABEL_3;
  if ( a4 || (v8 = v7, sub_9876C0((__int64 *)a2 + 1)) )
  {
LABEL_8:
    v10 = *((_DWORD *)a2 + 4);
    *(_DWORD *)(a1 + 8) = v10;
    if ( v10 > 0x40 )
    {
      sub_C43780(a1, (const void **)a2 + 1);
      v12 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 24) = v12;
      if ( v12 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)a1 = *((_QWORD *)a2 + 1);
      v11 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 24) = v11;
      if ( v11 <= 0x40 )
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
      sub_AADB10(a1, v8, 1);
    else
      sub_AADB10(a1, v8, 0);
    return a1;
  }
}
