// Function: sub_31087C0
// Address: 0x31087c0
//
__int64 __fastcall sub_31087C0(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // r13
  __int64 *v10; // rbx
  _BYTE *v11; // r12
  unsigned __int8 v12; // al

  v1 = *a1;
  if ( v1 == 40 )
  {
    v2 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v2 = -32;
    if ( v1 != 85 )
    {
      v2 = -96;
      if ( v1 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v3 = sub_BD2BC0((__int64)a1);
    v5 = v3 + v4;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v5 >> 4) )
        goto LABEL_9;
    }
    else
    {
      if ( !(unsigned int)((v5 - sub_BD2BC0((__int64)a1)) >> 4) )
        goto LABEL_9;
      if ( (a1[7] & 0x80u) != 0 )
      {
        v6 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v7 = sub_BD2BC0((__int64)a1);
        v2 -= 32LL * (unsigned int)(*(_DWORD *)(v7 + v8 - 4) - v6);
        goto LABEL_9;
      }
    }
    BUG();
  }
LABEL_9:
  v9 = (__int64 *)&a1[v2];
  v10 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  if ( v10 == v9 )
    return (unsigned __int8)sub_B49E20((__int64)a1) == 0 ? 22 : 24;
  while ( 1 )
  {
    v11 = (_BYTE *)*v10;
    v12 = *(_BYTE *)*v10;
    if ( v12 > 0x15u
      && v12 != 60
      && (v12 != 22
       || !(unsigned __int8)sub_B2BAE0(*v10)
       && !(unsigned __int8)sub_B2D6E0((__int64)v11)
       && !(unsigned __int8)sub_B2D720((__int64)v11))
      && *(_BYTE *)(*((_QWORD *)v11 + 1) + 8LL) == 14 )
    {
      break;
    }
    v10 += 4;
    if ( v9 == v10 )
      return (unsigned __int8)sub_B49E20((__int64)a1) == 0 ? 22 : 24;
  }
  return (unsigned __int8)sub_B49E20((__int64)a1) == 0 ? 21 : 23;
}
