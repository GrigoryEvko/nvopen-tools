// Function: sub_BDDCF0
// Address: 0xbddcf0
//
unsigned __int64 __fastcall sub_BDDCF0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rdx
  const char *v6; // r14
  __int64 v7; // rcx
  __int64 v8; // r15
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  _QWORD v15[4]; // [rsp+0h] [rbp-50h] BYREF
  char v16; // [rsp+20h] [rbp-30h]
  char v17; // [rsp+21h] [rbp-2Fh]

  sub_BDA950(a1, (const char *)a2);
  if ( (unsigned __int16)sub_AF18C0(a2) == 52 )
  {
    result = *(unsigned __int8 *)(a2 - 16);
    if ( (result & 2) != 0 )
    {
      v5 = *(_QWORD *)(a2 - 32);
      v6 = *(const char **)(v5 + 24);
      if ( v6 )
        goto LABEL_4;
    }
    else
    {
      result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
      v5 = a2 - 16 - result;
      v6 = *(const char **)(v5 + 24);
      if ( v6 )
      {
LABEL_4:
        result = *(unsigned __int8 *)v6;
        if ( (unsigned __int8)result > 0x24u || (v7 = 0x140000F000LL, !_bittest64(&v7, result)) )
        {
          v8 = *(_QWORD *)a1;
          v17 = 1;
          v15[0] = "invalid type ref";
          v16 = 3;
          if ( v8 )
          {
            sub_CA0E80(v15, v8);
            v9 = *(_BYTE **)(v8 + 32);
            if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
            {
              sub_CB5D20(v8, 10);
            }
            else
            {
              *(_QWORD *)(v8 + 32) = v9 + 1;
              *v9 = 10;
            }
            v10 = *(_QWORD *)a1;
            result = *(unsigned __int8 *)(a1 + 154);
            *(_BYTE *)(a1 + 153) = 1;
            *(_BYTE *)(a1 + 152) |= result;
            if ( !v10 )
              return result;
LABEL_24:
            sub_BD9900((__int64 *)a1, (const char *)a2);
            return (unsigned __int64)sub_BD9900((__int64 *)a1, v6);
          }
LABEL_22:
          result = *(unsigned __int8 *)(a1 + 154);
          *(_BYTE *)(a1 + 153) = 1;
          *(_BYTE *)(a1 + 152) |= result;
          return result;
        }
LABEL_13:
        v6 = *(const char **)(v5 + 48);
        if ( !v6 )
          return result;
        if ( *v6 == 13 )
          return result;
        v17 = 1;
        v15[0] = "invalid static data member declaration";
        v16 = 3;
        result = sub_BDD6D0((__int64 *)a1, (__int64)v15);
        if ( !*(_QWORD *)a1 )
          return result;
        goto LABEL_24;
      }
    }
    if ( !*(_BYTE *)(a2 + 21) )
      goto LABEL_13;
    v17 = 1;
    v15[0] = "missing global variable type";
    v16 = 3;
    result = sub_BDD6D0((__int64 *)a1, (__int64)v15);
    if ( *(_QWORD *)a1 )
      return (unsigned __int64)sub_BD9900((__int64 *)a1, (const char *)a2);
  }
  else
  {
    v11 = *(_QWORD *)a1;
    v17 = 1;
    v15[0] = "invalid tag";
    v16 = 3;
    if ( !v11 )
      goto LABEL_22;
    sub_CA0E80(v15, v11);
    v12 = *(_BYTE **)(v11 + 32);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
    {
      sub_CB5D20(v11, 10);
    }
    else
    {
      *(_QWORD *)(v11 + 32) = v12 + 1;
      *v12 = 10;
    }
    v13 = *(_QWORD *)a1;
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
    if ( v13 )
    {
      sub_A62C00((const char *)a2, v13, a1 + 16, *(_QWORD *)(a1 + 8));
      v14 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        return sub_CB5D20(v14, 10);
      }
      else
      {
        *(_QWORD *)(v14 + 32) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  return result;
}
