// Function: sub_100C200
// Address: 0x100c200
//
bool __fastcall sub_100C200(__int64 a1, int a2, unsigned __int8 *a3)
{
  unsigned __int8 *v4; // rcx
  int v5; // eax
  int v6; // eax
  unsigned __int8 *v7; // rcx
  __int64 v8; // rcx
  unsigned __int8 *v9; // rdx
  int v10; // eax
  int v11; // eax
  unsigned __int8 *v12; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
  v5 = *v4;
  if ( (unsigned __int8)v5 > 0x1Cu )
  {
    v6 = v5 - 29;
  }
  else
  {
    if ( (_BYTE)v5 != 5 )
      return 0;
    v6 = *((unsigned __int16 *)v4 + 1);
  }
  if ( v6 != 47 )
    return 0;
  v7 = (v4[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v4 - 1) : &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v8 = *(_QWORD *)v7;
  if ( !v8 )
    return 0;
  **(_QWORD **)a1 = v8;
  v9 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  v10 = *v9;
  if ( (unsigned __int8)v10 <= 0x1Cu )
  {
    if ( (_BYTE)v10 == 5 )
    {
      v11 = *((unsigned __int16 *)v9 + 1);
      goto LABEL_13;
    }
    return 0;
  }
  v11 = v10 - 29;
LABEL_13:
  if ( v11 != 47 )
    return 0;
  if ( (v9[7] & 0x40) != 0 )
    v12 = (unsigned __int8 *)*((_QWORD *)v9 - 1);
  else
    v12 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
  return *(_QWORD *)v12 == *(_QWORD *)(a1 + 8);
}
