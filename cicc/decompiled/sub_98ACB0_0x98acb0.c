// Function: sub_98ACB0
// Address: 0x98acb0
//
unsigned __int8 *__fastcall sub_98ACB0(unsigned __int8 *a1, unsigned int a2)
{
  __int64 v3; // rbx
  int v4; // r13d
  int v5; // eax
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int16 v9; // ax
  __int64 *v10; // rdx

  v3 = 0x8000000000041LL;
  v4 = 0;
LABEL_2:
  v5 = *a1;
  if ( (unsigned __int8)v5 <= 0x1Cu )
  {
    while ( (_BYTE)v5 != 5 )
    {
      if ( (_BYTE)v5 != 1 || (unsigned __int8)sub_B2F6B0(a1) )
        return a1;
      a1 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
LABEL_12:
      if ( a2 > ++v4 )
        goto LABEL_2;
      if ( a2 )
        return a1;
      v5 = *a1;
      if ( (unsigned __int8)v5 > 0x1Cu )
        goto LABEL_3;
    }
    v9 = *((_WORD *)a1 + 1);
    if ( v9 != 34 )
    {
      if ( (unsigned __int16)(v9 - 49) > 1u )
        return a1;
LABEL_18:
      if ( (a1[7] & 0x40) != 0 )
        v10 = (__int64 *)*((_QWORD *)a1 - 1);
      else
        v10 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v7 = *v10;
      if ( *(_BYTE *)(*(_QWORD *)(*v10 + 8) + 8LL) == 14 )
        goto LABEL_11;
      return a1;
    }
LABEL_10:
    v7 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) != 14 )
      return a1;
    goto LABEL_11;
  }
LABEL_3:
  if ( (_BYTE)v5 == 63 )
    goto LABEL_10;
  if ( (unsigned __int8)(v5 - 78) <= 1u )
    goto LABEL_18;
  if ( (_BYTE)v5 != 84 )
  {
    v6 = (unsigned int)(v5 - 34);
    if ( (unsigned __int8)v6 > 0x33u )
      return a1;
    if ( !_bittest64(&v3, v6) )
      return a1;
    v7 = sub_98AC40((__int64)a1, 0);
    if ( !v7 )
      return a1;
LABEL_11:
    a1 = (unsigned __int8 *)v7;
    goto LABEL_12;
  }
  if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) == 1 )
  {
    a1 = (unsigned __int8 *)**((_QWORD **)a1 - 1);
    goto LABEL_12;
  }
  return a1;
}
