// Function: sub_35DE150
// Address: 0x35de150
//
__int64 __fastcall sub_35DE150(_DWORD *a1, char *a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  unsigned int v7; // r8d
  unsigned int v8; // eax
  unsigned int v9; // r8d
  unsigned int v10; // r8d

  v5 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v5 == 62 )
  {
    LOBYTE(v7) = (unsigned int)sub_BCB060(*(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL)) <= *a1;
    return v7;
  }
  else if ( v5 == 30 )
  {
    if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) == 0 )
      BUG();
    LOBYTE(v9) = (unsigned int)sub_BCB060(*(_QWORD *)(*(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] + 8LL)) <= *a1;
    return v9;
  }
  else if ( v5 == 68 )
  {
    LOBYTE(v10) = (unsigned int)sub_BCB060(*((_QWORD *)a2 + 1)) > *a1;
    return v10;
  }
  else
  {
    if ( v5 == 32 )
    {
      LOBYTE(a5) = (unsigned int)sub_BCB060(*(_QWORD *)(**((_QWORD **)a2 - 1) + 8LL)) < *a1;
    }
    else
    {
      LOBYTE(a5) = v5 == 85;
      if ( v5 == 82 )
      {
        LOBYTE(v8) = sub_B532B0(*((_WORD *)a2 + 1) & 0x3F);
        a5 = v8;
        if ( !(_BYTE)v8 )
          LOBYTE(a5) = (unsigned int)sub_BCB060(*(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL)) < *a1;
      }
    }
    return a5;
  }
}
