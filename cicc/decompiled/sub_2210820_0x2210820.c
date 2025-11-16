// Function: sub_2210820
// Address: 0x2210820
//
__int64 __fastcall sub_2210820(__int64 *a1, char **a2, unsigned __int64 a3, char a4, int a5)
{
  __int64 v5; // r14
  __int64 v6; // r13
  char *v10; // rdx
  int v11; // ebx
  __int16 v12; // r8
  unsigned int v13; // eax
  char *v14; // rcx
  unsigned int v16; // r8d
  __int16 v17; // ax
  __int16 v18; // r8

  v5 = *a1;
  v6 = a1[1];
  if ( (a4 & 4) != 0 && (unsigned __int64)(v6 - v5) > 2 && *(_WORD *)v5 == 0xBBEF && *(_BYTE *)(v5 + 2) == 0xBF )
  {
    v5 += 3;
    *a1 = v5;
  }
  if ( v5 != v6 )
  {
    v10 = a2[1];
    v11 = a4 & 1;
    while ( v10 != *a2 )
    {
      v13 = sub_220F920((__int64)a1, a3);
      if ( v13 == -2 )
        return (unsigned int)(a5 + 1);
      if ( v13 > a3 )
        return 2;
      v10 = a2[1];
      v14 = *a2;
      if ( v13 <= 0xFFFF )
      {
        if ( v10 == v14 )
          goto LABEL_13;
        v6 = a1[1];
        v5 = *a1;
        v12 = __ROL2__(v13, 8);
        if ( !v11 )
          LOWORD(v13) = v12;
        *a2 = v14 + 2;
        *(_WORD *)v14 = v13;
        if ( v6 == v5 )
          return 0;
      }
      else
      {
        if ( (unsigned __int64)(v10 - v14) <= 2 )
        {
LABEL_13:
          *a1 = v5;
          a1[1] = v6;
          return 1;
        }
        v16 = v13 >> 10;
        v17 = (v13 & 0x3FF) - 9216;
        v18 = v16 - 10304;
        if ( !v11 )
        {
          v18 = __ROL2__(v18, 8);
          v17 = __ROL2__(v17, 8);
        }
        v6 = a1[1];
        v5 = *a1;
        *(_WORD *)v14 = v18;
        *a2 = v14 + 4;
        *((_WORD *)v14 + 1) = v17;
        if ( v6 == v5 )
          return 0;
      }
    }
  }
  return 0;
}
