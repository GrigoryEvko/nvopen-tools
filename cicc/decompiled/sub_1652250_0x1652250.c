// Function: sub_1652250
// Address: 0x1652250
//
unsigned __int64 __fastcall sub_1652250(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned __int8 *v5; // r14
  unsigned __int64 result; // rax
  unsigned __int8 *v7; // r15
  __int64 v8; // r14
  _BYTE *v9; // rax
  char v10; // dl
  __int64 v11; // r15
  _BYTE *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdi
  _QWORD v16[2]; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+10h] [rbp-30h]
  char v18; // [rsp+11h] [rbp-2Fh]

  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(unsigned __int8 **)(a2 - 8 * v4);
  if ( !v5 )
  {
LABEL_12:
    v11 = *(_QWORD *)a1;
    v18 = 1;
    v16[0] = "location requires a valid scope";
    v17 = 3;
    if ( !v11 )
      goto LABEL_21;
    sub_16E2CE0(v16, v11);
    v12 = *(_BYTE **)(v11 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
    {
      sub_16E7DE0(v11, 10);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 10;
    }
    v13 = *(_QWORD *)a1;
    result = *(unsigned __int8 *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= result;
    if ( v13 )
    {
      sub_15562E0((unsigned __int8 *)a2, v13, a1 + 16, *(_QWORD *)(a1 + 8));
      v14 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
        result = sub_16E7DE0(v14, 10);
      }
      else
      {
        *(_QWORD *)(v14 + 24) = result + 1;
        *(_BYTE *)result = 10;
      }
      if ( v5 )
      {
        sub_15562E0(v5, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
        v15 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          return sub_16E7DE0(v15, 10);
        }
        else
        {
          *(_QWORD *)(v15 + 24) = result + 1;
          *(_BYTE *)result = 10;
        }
      }
    }
    return result;
  }
  result = *v5;
  if ( (_DWORD)result != 17 )
  {
    result = (unsigned int)(result - 18);
    if ( (unsigned int)result <= 1 )
    {
      if ( (_DWORD)v4 != 2 )
        return result;
      v7 = *(unsigned __int8 **)(a2 - 8);
      if ( !v7 || *v7 == 5 )
        return result;
LABEL_7:
      v8 = *(_QWORD *)a1;
      v18 = 1;
      v16[0] = "inlined-at should be a location";
      v17 = 3;
      if ( v8 )
      {
        sub_16E2CE0(v16, v8);
        v9 = *(_BYTE **)(v8 + 24);
        if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
        {
          sub_16E7DE0(v8, 10);
        }
        else
        {
          *(_QWORD *)(v8 + 24) = v9 + 1;
          *v9 = 10;
        }
        result = *(_QWORD *)a1;
        v10 = *(_BYTE *)(a1 + 74);
        *(_BYTE *)(a1 + 73) = 1;
        *(_BYTE *)(a1 + 72) |= v10;
        if ( result )
        {
          sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
          return (unsigned __int64)sub_164ED40((__int64 *)a1, v7);
        }
        return result;
      }
LABEL_21:
      result = *(unsigned __int8 *)(a1 + 74);
      *(_BYTE *)(a1 + 72) |= result;
      *(_BYTE *)(a1 + 73) = 1;
      return result;
    }
    goto LABEL_12;
  }
  if ( (_DWORD)v4 == 2 )
  {
    v7 = *(unsigned __int8 **)(a2 - 8);
    if ( v7 )
    {
      if ( *v7 != 5 )
        goto LABEL_7;
    }
  }
  if ( (v5[40] & 8) == 0 )
  {
    v18 = 1;
    v16[0] = "scope points into the type hierarchy";
    v17 = 3;
    result = sub_16521E0((__int64 *)a1, (__int64)v16);
    if ( *(_QWORD *)a1 )
      return (unsigned __int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
  }
  return result;
}
