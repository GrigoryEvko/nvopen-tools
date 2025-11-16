// Function: sub_2588100
// Address: 0x2588100
//
char __fastcall sub_2588100(__int64 **a1, __int64 *a2, _BYTE *a3)
{
  unsigned __int8 *v5; // r13
  __int64 v6; // rdx
  int v7; // ecx
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  char v18; // r15
  __int16 v19; // ax
  char result; // al
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdx
  bool v26; // [rsp+1Fh] [rbp-51h] BYREF
  __int64 v27[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 (__fastcall *v28)(__int64 *, __int64 *, int); // [rsp+30h] [rbp-40h]
  bool (__fastcall *v29)(_QWORD *, __int64); // [rsp+38h] [rbp-38h]

  v5 = (unsigned __int8 *)a2[3];
  if ( v5 == (unsigned __int8 *)sub_2509740(*a1 + 9) && (*((_DWORD *)v5 + 1) & 0x7FFFFFF) == 1 )
    return 1;
  v6 = *a1[1];
  if ( v6 )
  {
    v7 = *v5;
    if ( (unsigned __int8)(v7 - 34) > 0x33u
      || (v8 = 0x8000000000041LL, !_bittest64(&v8, (unsigned int)(v7 - 34)))
      || a2 < (__int64 *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)] )
    {
LABEL_19:
      v16 = (__int64)*a1;
      v27[0] = v6;
      v29 = sub_2535010;
      v28 = sub_2535A80;
      v17 = sub_2509740((_QWORD *)(v16 + 72));
      v18 = sub_2529340((__int64)a1[2], (__int64)v5, v17, v16, 0, (__int64)v27);
      if ( v28 )
        v28(v27, v27, 3);
      if ( !v18 )
        return 1;
      goto LABEL_22;
    }
    if ( v7 == 40 )
    {
      v9 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v5);
    }
    else
    {
      v9 = -32;
      if ( v7 != 85 )
      {
        v9 = -96;
        if ( v7 != 34 )
          BUG();
      }
    }
    if ( (v5[7] & 0x80u) != 0 )
    {
      v10 = sub_BD2BC0((__int64)v5);
      v12 = v10 + v11;
      if ( (v5[7] & 0x80u) == 0 )
      {
        if ( !(unsigned int)(v12 >> 4) )
          goto LABEL_17;
      }
      else
      {
        if ( !(unsigned int)((v12 - sub_BD2BC0((__int64)v5)) >> 4) )
          goto LABEL_17;
        if ( (v5[7] & 0x80u) != 0 )
        {
          v13 = *(_DWORD *)(sub_BD2BC0((__int64)v5) + 8);
          if ( (v5[7] & 0x80u) == 0 )
            BUG();
          v14 = sub_BD2BC0((__int64)v5);
          v9 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
          goto LABEL_17;
        }
      }
      BUG();
    }
LABEL_17:
    if ( a2 < (__int64 *)&v5[v9] )
    {
      v21 = sub_254C9B0((__int64)v5, ((char *)a2 - (char *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)]) >> 5);
      v22 = (__int64)a1[2];
      v23 = (__int64)*a1;
      v27[1] = v24;
      v27[0] = v21;
      result = sub_2588040(v22, v23, v27, 1, &v26, 0, 0);
      if ( result )
        return result;
      v6 = *a1[1];
    }
    else
    {
      v6 = *a1[1];
    }
    goto LABEL_19;
  }
LABEL_22:
  v19 = sub_D139D0(
          a2,
          0,
          (unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, __int64))sub_258EF20,
          (__int64)a1[3]);
  if ( !v19 )
    return 1;
  result = HIBYTE(v19) != 0 && (_BYTE)v19 == 0;
  if ( result )
    *a3 = 1;
  return result;
}
