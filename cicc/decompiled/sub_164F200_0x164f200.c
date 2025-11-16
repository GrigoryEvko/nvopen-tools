// Function: sub_164F200
// Address: 0x164f200
//
unsigned __int64 __fastcall sub_164F200(__int64 a1, __int64 a2)
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
  __int64 v13; // rdi
  unsigned __int8 *v14; // r14
  __int64 v15; // r15
  _BYTE *v16; // rax
  __int64 v17; // rdx
  _QWORD v18[2]; // [rsp+0h] [rbp-40h] BYREF
  char v19; // [rsp+10h] [rbp-30h]
  char v20; // [rsp+11h] [rbp-2Fh]

  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(unsigned __int8 **)(a2 - 8 * v4);
  if ( v5 )
  {
    v6 = *v5;
    if ( *v5 > 0x15u )
    {
      if ( (unsigned __int8)(v6 - 31) > 2u )
      {
LABEL_4:
        v7 = *(_QWORD *)a1;
        v20 = 1;
        v18[0] = "invalid scope";
        v19 = 3;
        if ( v7 )
        {
          sub_16E2CE0(v18, v7);
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
            sub_15562E0(v5, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
            v13 = *(_QWORD *)a1;
            result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
            if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
              return sub_16E7DE0(v13, 10);
            }
            else
            {
              *(_QWORD *)(v13 + 24) = result + 1;
              *(_BYTE *)result = 10;
            }
          }
          return result;
        }
LABEL_21:
        result = *(unsigned __int8 *)(a1 + 74);
        *(_BYTE *)(a1 + 73) = 1;
        *(_BYTE *)(a1 + 72) |= result;
        return result;
      }
    }
    else if ( v6 <= 0xAu )
    {
      goto LABEL_4;
    }
  }
  result = 2 - v4;
  v14 = *(unsigned __int8 **)(a2 + 8 * (2 - v4));
  if ( !v14 || *v14 == 15 )
    return result;
  v15 = *(_QWORD *)a1;
  v20 = 1;
  v18[0] = "invalid file";
  v19 = 3;
  if ( !v15 )
    goto LABEL_21;
  sub_16E2CE0(v18, v15);
  v16 = *(_BYTE **)(v15 + 24);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
  {
    sub_16E7DE0(v15, 10);
  }
  else
  {
    *(_QWORD *)(v15 + 24) = v16 + 1;
    *v16 = 10;
  }
  v17 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(a1 + 74);
  *(_BYTE *)(a1 + 73) = 1;
  *(_BYTE *)(a1 + 72) |= result;
  if ( v17 )
  {
    sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
    return (unsigned __int64)sub_164ED40((__int64 *)a1, v14);
  }
  return result;
}
