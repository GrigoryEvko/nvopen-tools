// Function: sub_2624C30
// Address: 0x2624c30
//
__int64 __fastcall sub_2624C30(__int64 a1, __int64 a2, unsigned __int8 *a3, _QWORD *a4)
{
  unsigned int v4; // r15d
  int v7; // eax
  __int64 *v8; // rsi
  __int64 *i; // rcx
  _QWORD *v10; // rdx
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  _QWORD *v14; // rax
  int v16; // eax
  unsigned __int8 *v17; // r14
  unsigned __int64 v18; // rax
  unsigned __int8 *v19; // rdx
  unsigned __int8 *v20; // r14
  __int64 *v21; // [rsp+0h] [rbp-50h] BYREF
  __int64 v22; // [rsp+8h] [rbp-48h]
  _BYTE v23[64]; // [rsp+10h] [rbp-40h] BYREF

  while ( 1 )
  {
    v7 = *a3;
    LOBYTE(v4) = (_BYTE)v7 == 0 || (unsigned __int8)(v7 - 2) <= 1u;
    if ( (_BYTE)v4 )
      break;
    while ( 1 )
    {
      if ( (_BYTE)v7 == 63 )
      {
LABEL_26:
        LODWORD(v22) = sub_AE2980(a2, 0)[3];
        if ( (unsigned int)v22 > 0x40 )
          sub_C43690((__int64)&v21, 0, 0);
        else
          v21 = 0;
        v4 = sub_BB6360((__int64)a3, a2, (__int64)&v21, 0, 0);
        if ( (_BYTE)v4 )
        {
          v18 = (unsigned __int64)v21;
          if ( (unsigned int)v22 > 0x40 )
            v18 = *v21;
          v4 = sub_2624C30(a1, a2, *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)], (char *)a4 + v18);
        }
        if ( (unsigned int)v22 > 0x40 && v21 )
          j_j___libc_free_0_0((unsigned __int64)v21);
        return v4;
      }
      if ( (_BYTE)v7 == 5 )
      {
        v16 = *((unsigned __int16 *)a3 + 1);
        if ( (_WORD)v16 == 34 )
          goto LABEL_26;
      }
      else
      {
        if ( (unsigned __int8)v7 <= 0x1Cu )
          return v4;
        v16 = v7 - 29;
      }
      if ( v16 != 49 )
        break;
      if ( (a3[7] & 0x40) != 0 )
        v17 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
      else
        v17 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
      a3 = *(unsigned __int8 **)v17;
      v7 = *a3;
      if ( (unsigned __int8)(v7 - 2) <= 1u || !(_BYTE)v7 )
        goto LABEL_2;
    }
    if ( v16 != 57 )
      return v4;
    v19 = (a3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a3 - 1) : &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    if ( !(unsigned __int8)sub_2624C30(a1, a2, *((_QWORD *)v19 + 4), a4) )
      return v4;
    if ( (a3[7] & 0x40) != 0 )
      v20 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    else
      v20 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    a3 = (unsigned __int8 *)*((_QWORD *)v20 + 8);
  }
LABEL_2:
  v21 = (__int64 *)v23;
  v22 = 0x200000000LL;
  sub_B91D10((__int64)a3, 19, (__int64)&v21);
  v8 = &v21[(unsigned int)v22];
  for ( i = v21; v8 != i; ++i )
  {
    v11 = *i;
    v12 = *(_BYTE *)(*i - 16);
    if ( (v12 & 2) != 0 )
    {
      v10 = *(_QWORD **)(v11 - 32);
      if ( a1 != v10[1] )
        continue;
    }
    else
    {
      v10 = (_QWORD *)(-16 - 8LL * ((v12 >> 2) & 0xF) + v11);
      if ( a1 != v10[1] )
        continue;
    }
    v13 = *(_QWORD *)(*v10 + 136LL);
    v14 = *(_QWORD **)(v13 + 24);
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
      v14 = (_QWORD *)*v14;
    if ( a4 == v14 )
    {
      v4 = 1;
      if ( v21 != (__int64 *)v23 )
        goto LABEL_12;
      return v4;
    }
  }
  v4 = 0;
  if ( v21 != (__int64 *)v23 )
LABEL_12:
    _libc_free((unsigned __int64)v21);
  return v4;
}
