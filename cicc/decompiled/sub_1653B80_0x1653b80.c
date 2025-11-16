// Function: sub_1653B80
// Address: 0x1653b80
//
unsigned __int64 __fastcall sub_1653B80(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned __int8 *v5; // r14
  unsigned __int8 v6; // al
  __int64 v7; // r15
  _BYTE *v8; // rax
  unsigned __int64 result; // rax
  char v10; // dl
  unsigned __int8 v11; // al
  const char *v12; // rax
  __int64 v13; // r14
  int v14; // eax
  _BYTE *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  _QWORD v18[2]; // [rsp+0h] [rbp-40h] BYREF
  char v19; // [rsp+10h] [rbp-30h]
  char v20; // [rsp+11h] [rbp-2Fh]

  sub_164F0A0((__int64)a1, a2);
  if ( *(_WORD *)(a2 + 2) != 1 )
  {
    v13 = *a1;
    v20 = 1;
    v18[0] = "invalid tag";
    v19 = 3;
    if ( v13 )
    {
      sub_16E2CE0(v18, v13);
      v15 = *(_BYTE **)(v13 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v13 + 16) )
      {
        sub_16E7DE0(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 24) = v15 + 1;
        *v15 = 10;
      }
      v16 = *a1;
      result = *((unsigned __int8 *)a1 + 74);
      *((_BYTE *)a1 + 73) = 1;
      *((_BYTE *)a1 + 72) |= result;
      if ( v16 )
      {
        sub_15562E0((unsigned __int8 *)a2, v16, (__int64)(a1 + 2), a1[1]);
        v17 = *a1;
        result = *(_QWORD *)(*a1 + 24);
        if ( result >= *(_QWORD *)(*a1 + 16) )
        {
          return sub_16E7DE0(v17, 10);
        }
        else
        {
          *(_QWORD *)(v17 + 24) = result + 1;
          *(_BYTE *)result = 10;
        }
      }
      return result;
    }
    goto LABEL_23;
  }
  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(unsigned __int8 **)(a2 + 8 * (1 - v4));
  if ( v5 )
  {
    v6 = *v5;
    if ( *v5 > 0x15u )
    {
      if ( (unsigned __int8)(v6 - 31) > 2u )
      {
LABEL_5:
        v7 = *a1;
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
          result = *a1;
          v10 = *((_BYTE *)a1 + 74);
          *((_BYTE *)a1 + 73) = 1;
          *((_BYTE *)a1 + 72) |= v10;
          if ( result )
            goto LABEL_9;
          return result;
        }
LABEL_23:
        result = *((unsigned __int8 *)a1 + 74);
        *((_BYTE *)a1 + 73) = 1;
        *((_BYTE *)a1 + 72) |= result;
        return result;
      }
    }
    else if ( v6 <= 0xAu )
    {
      goto LABEL_5;
    }
  }
  v5 = *(unsigned __int8 **)(a2 + 8 * (3 - v4));
  if ( !v5 )
    goto LABEL_19;
  v11 = *v5;
  if ( *v5 > 0xEu )
  {
    if ( (unsigned __int8)(v11 - 32) > 1u )
      goto LABEL_15;
LABEL_19:
    v5 = *(unsigned __int8 **)(a2 + 8 * (4 - v4));
    if ( !v5 || *v5 == 4 )
    {
      v14 = *(_DWORD *)(a2 + 28);
      if ( (v14 & 0x6000) == 0x6000 || (result = v14 & 0xC00000, (_DWORD)result == 12582912) )
      {
        v20 = 1;
        v18[0] = "invalid reference flags";
        v19 = 3;
        result = sub_16521E0(a1, (__int64)v18);
        if ( *a1 )
          return (unsigned __int64)sub_164ED40(a1, (unsigned __int8 *)a2);
      }
      return result;
    }
    v20 = 1;
    v12 = "invalid composite elements";
    goto LABEL_16;
  }
  if ( v11 > 0xAu )
    goto LABEL_19;
LABEL_15:
  v20 = 1;
  v12 = "invalid base type";
LABEL_16:
  v18[0] = v12;
  v19 = 3;
  result = sub_16521E0(a1, (__int64)v18);
  if ( *a1 )
  {
LABEL_9:
    sub_164ED40(a1, (unsigned __int8 *)a2);
    return (unsigned __int64)sub_164ED40(a1, v5);
  }
  return result;
}
