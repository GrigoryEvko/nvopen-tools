// Function: sub_31AF060
// Address: 0x31af060
//
__int64 __fastcall sub_31AF060(__int64 a1, unsigned __int8 a2)
{
  __int64 v3; // r13
  __int64 v4; // rsi
  char v5; // r13
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r12d
  __int64 v9; // rcx
  __int64 v11; // rdx
  unsigned int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r14
  unsigned int v19; // r13d
  int v20; // eax
  unsigned int (__fastcall ***v21)(_QWORD); // rax
  __int64 v22; // r13

  v3 = sub_B2BE50(**(_QWORD **)(a1 + 64));
  if ( sub_B6EA50(v3)
    || (v22 = sub_B6F970(v3),
        (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v22 + 32LL))(
          v22,
          "loop-vectorize",
          14))
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v22 + 40LL))(
         v22,
         "loop-vectorize",
         14)
    || (v5 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v22 + 24LL))(
               v22,
               "loop-vectorize",
               14)) != 0 )
  {
    v4 = *(_QWORD *)a1;
    v5 = 1;
    v8 = sub_31A6EB0(a1, *(_QWORD *)a1, a2);
  }
  else
  {
    v4 = *(_QWORD *)a1;
    v8 = sub_31A6EB0(a1, *(_QWORD *)a1, a2);
    if ( !(_BYTE)v8 )
      return 0;
  }
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) == v9 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)a1 + 40LL) - *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( (unsigned int)(v11 >> 3) == 1 || (v12 = sub_31A9940(a1), (_BYTE)v12) )
    {
      v12 = sub_31AD8F0(a1, v4, v11, v9, v6, v7);
      if ( (_BYTE)v12 )
      {
        v12 = v8;
      }
      else if ( !v5 )
      {
        return 0;
      }
    }
    else
    {
      if ( !v5 )
        return 0;
      sub_31AD8F0(a1, v4, v11, v9, v6, v7);
    }
    v13 = sub_DEF8F0(*(_QWORD **)(a1 + 16));
    if ( sub_D96A50(v13) )
    {
      if ( sub_D46F00(*(_QWORD *)a1) )
      {
        v4 = 33;
        sub_2AB8760(
          (__int64)"Cannot vectorize uncountable loop",
          33,
          "Cannot vectorize uncountable loop",
          0x21u,
          (__int64)"UnsupportedUncountableLoop",
          26,
          *(__int64 **)(a1 + 64),
          *(_QWORD *)a1,
          0);
      }
      else
      {
        if ( (unsigned __int8)sub_31A6FE0(a1) )
          goto LABEL_16;
        if ( *(_BYTE *)(a1 + 664) )
          *(_BYTE *)(a1 + 664) = 0;
      }
      v12 = 0;
      if ( !v5 )
        return 0;
    }
LABEL_16:
    v8 = sub_31A77E0(a1);
    if ( (_BYTE)v8 )
    {
      v8 = v12;
    }
    else if ( !v5 )
    {
      return 0;
    }
    v18 = *(_QWORD *)(a1 + 416);
    v19 = qword_5035348;
    v20 = *(_DWORD *)(v18 + 40);
    if ( v20 == -1 )
    {
      if ( (unsigned __int8)sub_F6E590(*(_QWORD *)(v18 + 104), v4, v14, v15, v16, v17) )
      {
LABEL_21:
        v21 = (unsigned int (__fastcall ***)(_QWORD))sub_D9B120(*(_QWORD *)(a1 + 16));
        if ( (**v21)(v21) > v19 )
        {
          v8 = 0;
          sub_2AB8760(
            (__int64)"Too many SCEV checks needed",
            27,
            "Too many SCEV assumptions need to be made and checked at runtime",
            0x40u,
            (__int64)"TooManySCEVRunTimeChecks",
            24,
            *(__int64 **)(a1 + 64),
            *(_QWORD *)a1,
            0);
        }
        return v8;
      }
      v20 = *(_DWORD *)(v18 + 40);
    }
    if ( v20 == 1 )
      v19 = qword_5035268;
    goto LABEL_21;
  }
  if ( (unsigned __int8)sub_31AD010((__int64 **)a1) )
    return v8;
  sub_2AB8760(
    (__int64)"Unsupported outer loop",
    22,
    "Unsupported outer loop",
    0x16u,
    (__int64)"UnsupportedOuterLoop",
    20,
    *(__int64 **)(a1 + 64),
    *(_QWORD *)a1,
    0);
  return 0;
}
