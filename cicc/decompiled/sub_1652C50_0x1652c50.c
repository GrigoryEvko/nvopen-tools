// Function: sub_1652C50
// Address: 0x1652c50
//
unsigned __int64 __fastcall sub_1652C50(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned __int8 *v5; // r14
  unsigned __int8 v6; // al
  __int64 v7; // r15
  _BYTE *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 result; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdx
  unsigned __int8 *v16; // rdi
  __int64 v17; // r14
  unsigned __int8 *v18; // r14
  __int64 v19; // r15
  _BYTE *v20; // rax
  char v21; // dl
  _BYTE *v22; // rax
  __int64 v23; // rdi
  _QWORD v24[2]; // [rsp+0h] [rbp-40h] BYREF
  char v25; // [rsp+10h] [rbp-30h]
  char v26; // [rsp+11h] [rbp-2Fh]

  sub_164F200(a1, a2);
  if ( *(_WORD *)(a2 + 2) == 52 )
  {
    v4 = *(unsigned int *)(a2 + 8);
    v5 = *(unsigned __int8 **)(a2 + 8 * (3 - v4));
    if ( !v5 )
    {
      v26 = 1;
      v24[0] = "missing global variable type";
      v25 = 3;
      result = sub_16521E0((__int64 *)a1, (__int64)v24);
      if ( *(_QWORD *)a1 )
        return (unsigned __int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      return result;
    }
    v6 = *v5;
    if ( *v5 > 0xEu )
    {
      if ( (unsigned __int8)(v6 - 32) > 1u )
      {
LABEL_5:
        v7 = *(_QWORD *)a1;
        v26 = 1;
        v24[0] = "invalid type ref";
        v25 = 3;
        if ( v7 )
        {
          sub_16E2CE0(v24, v7);
          v8 = *(_BYTE **)(v7 + 24);
          if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 16) )
          {
            sub_16E7DE0(v7, 10);
          }
          else
          {
            *(_QWORD *)(v7 + 24) = v8 + 1;
            *v8 = 10;
          }
          v9 = *(_QWORD *)a1;
          result = *(unsigned __int8 *)(a1 + 74);
          *(_BYTE *)(a1 + 73) = 1;
          *(_BYTE *)(a1 + 72) |= result;
          if ( v9 )
          {
            sub_15562E0((unsigned __int8 *)a2, v9, a1 + 16, *(_QWORD *)(a1 + 8));
            v11 = *(_QWORD *)a1;
            v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v12 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
              sub_16E7DE0(v11, 10);
            }
            else
            {
              *(_QWORD *)(v11 + 24) = v12 + 1;
              *v12 = 10;
            }
            v13 = *(_QWORD *)(a1 + 8);
            v14 = *(_QWORD *)a1;
            v15 = a1 + 16;
            v16 = v5;
            goto LABEL_27;
          }
          return result;
        }
LABEL_13:
        result = *(unsigned __int8 *)(a1 + 74);
        *(_BYTE *)(a1 + 72) |= result;
        *(_BYTE *)(a1 + 73) = 1;
        return result;
      }
    }
    else if ( v6 <= 0xAu )
    {
      goto LABEL_5;
    }
    result = 6 - v4;
    v18 = *(unsigned __int8 **)(a2 + 8 * (6 - v4));
    if ( !v18 || *v18 == 12 )
      return result;
    v19 = *(_QWORD *)a1;
    v26 = 1;
    v24[0] = "invalid static data member declaration";
    v25 = 3;
    if ( v19 )
    {
      sub_16E2CE0(v24, v19);
      v20 = *(_BYTE **)(v19 + 24);
      if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 16) )
      {
        sub_16E7DE0(v19, 10);
      }
      else
      {
        *(_QWORD *)(v19 + 24) = v20 + 1;
        *v20 = 10;
      }
      result = *(_QWORD *)a1;
      v21 = *(_BYTE *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= v21;
      if ( result )
      {
        sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
        return (unsigned __int64)sub_164ED40((__int64 *)a1, v18);
      }
      return result;
    }
    goto LABEL_13;
  }
  v17 = *(_QWORD *)a1;
  v26 = 1;
  v24[0] = "invalid tag";
  v25 = 3;
  if ( !v17 )
    goto LABEL_13;
  sub_16E2CE0(v24, v17);
  v22 = *(_BYTE **)(v17 + 24);
  if ( (unsigned __int64)v22 >= *(_QWORD *)(v17 + 16) )
  {
    sub_16E7DE0(v17, 10);
  }
  else
  {
    *(_QWORD *)(v17 + 24) = v22 + 1;
    *v22 = 10;
  }
  v14 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(a1 + 74);
  *(_BYTE *)(a1 + 73) = 1;
  *(_BYTE *)(a1 + 72) |= result;
  if ( v14 )
  {
    v13 = *(_QWORD *)(a1 + 8);
    v15 = a1 + 16;
    v16 = (unsigned __int8 *)a2;
LABEL_27:
    sub_15562E0(v16, v14, v15, v13);
    v23 = *(_QWORD *)a1;
    result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
    {
      return sub_16E7DE0(v23, 10);
    }
    else
    {
      *(_QWORD *)(v23 + 24) = result + 1;
      *(_BYTE *)result = 10;
    }
  }
  return result;
}
