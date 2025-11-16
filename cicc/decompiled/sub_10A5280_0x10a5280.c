// Function: sub_10A5280
// Address: 0x10a5280
//
__int64 __fastcall sub_10A5280(_QWORD **a1, unsigned __int8 *a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v5; // edx
  int v6; // eax
  unsigned __int8 *v7; // rsi
  unsigned __int8 *v8; // rcx
  int v9; // edx
  unsigned __int8 *v10; // rcx

  v2 = *a2;
  if ( (_BYTE)v2 == 68 )
  {
    v8 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v9 = *v8;
    if ( (unsigned __int8)v9 > 0x1Cu )
    {
      v5 = v9 - 29;
    }
    else
    {
      if ( (_BYTE)v9 != 5 )
        goto LABEL_7;
      v5 = *((unsigned __int16 *)v8 + 1);
    }
    if ( v5 == 47 )
    {
      v10 = (v8[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v8 - 1) : &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
      if ( *(_QWORD *)v10 )
      {
        v3 = 1;
        **a1 = *(_QWORD *)v10;
        return v3;
      }
    }
    goto LABEL_7;
  }
  if ( (unsigned __int8)v2 > 0x1Cu )
  {
LABEL_7:
    v6 = v2 - 29;
    goto LABEL_8;
  }
  v3 = 0;
  if ( (_BYTE)v2 != 5 )
    return v3;
  v6 = *((unsigned __int16 *)a2 + 1);
LABEL_8:
  v3 = 0;
  if ( v6 == 47 )
  {
    if ( (a2[7] & 0x40) != 0 )
      v7 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v7 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v3 = 0;
    if ( *(_QWORD *)v7 )
    {
      v3 = 1;
      *a1[1] = *(_QWORD *)v7;
    }
  }
  return v3;
}
