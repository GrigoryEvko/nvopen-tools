// Function: sub_17587F0
// Address: 0x17587f0
//
_QWORD *__fastcall sub_17587F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r15
  __int16 v5; // r13
  __int64 v6; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *result; // rax
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned int i; // edx
  __int64 v17; // rax
  char v18; // si
  unsigned int v19; // edx
  bool v20; // al
  int v21; // [rsp+4h] [rbp-7Ch]
  _QWORD *v22; // [rsp+8h] [rbp-78h]
  int v23; // [rsp+8h] [rbp-78h]
  unsigned int v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26[3]; // [rsp+18h] [rbp-68h] BYREF
  char v27[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v28; // [rsp+40h] [rbp-40h]

  v4 = *(_BYTE **)(a2 - 24);
  if ( v4[16] > 0x10u )
    return 0;
  v5 = *(_WORD *)(a2 + 18);
  v6 = *(_QWORD *)(a2 - 48);
  if ( !sub_1593BB0(*(_QWORD *)(a2 - 24), a2, a3, a4) )
  {
    if ( v4[16] == 13 )
    {
      if ( *((_DWORD *)v4 + 8) <= 0x40u )
      {
        if ( *((_QWORD *)v4 + 3) )
          return 0;
      }
      else
      {
        v23 = *((_DWORD *)v4 + 8);
        if ( v23 != (unsigned int)sub_16A57B0((__int64)(v4 + 24)) )
          return 0;
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) != 16 )
        return 0;
      v14 = sub_15A1020(v4, a2, v8, v9);
      if ( v14 && *(_BYTE *)(v14 + 16) == 13 )
      {
        if ( !sub_13D01C0(v14 + 24) )
          return 0;
      }
      else
      {
        v21 = *(_QWORD *)(*(_QWORD *)v4 + 32LL);
        if ( v21 )
        {
          for ( i = 0; i != v21; i = v19 + 1 )
          {
            v24 = i;
            v17 = sub_15A0A60((__int64)v4, i);
            if ( !v17 )
              return 0;
            v18 = *(_BYTE *)(v17 + 16);
            v19 = v24;
            if ( v18 != 9 )
            {
              if ( v18 != 13 )
                return 0;
              v20 = sub_13D01C0(v17 + 24);
              v19 = v24;
              if ( !v20 )
                return 0;
            }
          }
        }
      }
    }
  }
  if ( (v5 & 0x7FFF) != 0x26 || (unsigned int)sub_14B2890(v6, &v25, v26, 0, 0) != 1 )
    return 0;
  if ( (unsigned __int8)sub_14C27E0(v25, a1[333], 0, a1[330], a2, a1[332]) )
  {
    v11 = *(_QWORD *)(a2 - 24);
    v28 = 257;
    result = sub_1648A60(56, 2u);
    if ( result )
    {
      v12 = v26[0];
      v13 = v11;
LABEL_10:
      v22 = result;
      sub_17582E0((__int64)result, 38, v12, v13, (__int64)v27);
      return v22;
    }
    return result;
  }
  if ( !(unsigned __int8)sub_14C27E0(v26[0], a1[333], 0, a1[330], a2, a1[332]) )
    return 0;
  v15 = *(_QWORD *)(a2 - 24);
  v28 = 257;
  result = sub_1648A60(56, 2u);
  if ( result )
  {
    v12 = v25;
    v13 = v15;
    goto LABEL_10;
  }
  return result;
}
