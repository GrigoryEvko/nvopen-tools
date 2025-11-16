// Function: sub_1652730
// Address: 0x1652730
//
unsigned __int64 __fastcall sub_1652730(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // rdx
  unsigned __int8 **v6; // rax
  unsigned __int8 *v7; // r15
  unsigned __int8 v8; // dl
  __int64 v9; // rbx
  _BYTE *v10; // rax
  __int64 v11; // rdx
  unsigned __int64 result; // rax
  int v13; // eax
  __int64 v14; // r14
  _BYTE *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r15
  _BYTE *v19; // rax
  char v20; // dl
  _QWORD v21[2]; // [rsp+0h] [rbp-50h] BYREF
  char v22; // [rsp+10h] [rbp-40h]
  char v23; // [rsp+11h] [rbp-3Fh]

  if ( *(_WORD *)(a2 + 2) == 21 )
  {
    v4 = *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)));
    if ( !v4 )
    {
LABEL_15:
      v13 = *(_DWORD *)(a2 + 28);
      if ( (v13 & 0x6000) == 0x6000 || (result = v13 & 0xC00000, (_DWORD)result == 12582912) )
      {
        v23 = 1;
        v21[0] = "invalid reference flags";
        v22 = 3;
        result = sub_16521E0((__int64 *)a1, (__int64)v21);
        if ( *(_QWORD *)a1 )
          return (unsigned __int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      }
      return result;
    }
    if ( *(_BYTE *)v4 == 4 )
    {
      v5 = 8LL * *(unsigned int *)(v4 + 8);
      v6 = (unsigned __int8 **)(v4 - v5);
      if ( v4 != v4 - v5 )
      {
        while ( 1 )
        {
          v7 = *v6;
          if ( *v6 )
          {
            v8 = *v7;
            if ( *v7 > 0xEu )
            {
              if ( (unsigned __int8)(v8 - 32) > 1u )
              {
LABEL_8:
                v9 = *(_QWORD *)a1;
                v23 = 1;
                v21[0] = "invalid subroutine type ref";
                v22 = 3;
                if ( v9 )
                {
                  sub_16E2CE0(v21, v9);
                  v10 = *(_BYTE **)(v9 + 24);
                  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
                  {
                    sub_16E7DE0(v9, 10);
                  }
                  else
                  {
                    *(_QWORD *)(v9 + 24) = v10 + 1;
                    *v10 = 10;
                  }
                  v11 = *(_QWORD *)a1;
                  result = *(unsigned __int8 *)(a1 + 74);
                  *(_BYTE *)(a1 + 73) = 1;
                  *(_BYTE *)(a1 + 72) |= result;
                  if ( v11 )
                  {
                    sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
                    sub_164ED40((__int64 *)a1, (unsigned __int8 *)v4);
                    return (unsigned __int64)sub_164ED40((__int64 *)a1, v7);
                  }
                  return result;
                }
LABEL_20:
                result = *(unsigned __int8 *)(a1 + 74);
                *(_BYTE *)(a1 + 72) |= result;
                *(_BYTE *)(a1 + 73) = 1;
                return result;
              }
            }
            else if ( v8 <= 0xAu )
            {
              goto LABEL_8;
            }
          }
          if ( (unsigned __int8 **)v4 == ++v6 )
            goto LABEL_15;
        }
      }
      goto LABEL_15;
    }
    v18 = *(_QWORD *)a1;
    v23 = 1;
    v21[0] = "invalid composite elements";
    v22 = 3;
    if ( !v18 )
      goto LABEL_20;
    sub_16E2CE0(v21, v18);
    v19 = *(_BYTE **)(v18 + 24);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 16) )
    {
      sub_16E7DE0(v18, 10);
    }
    else
    {
      *(_QWORD *)(v18 + 24) = v19 + 1;
      *v19 = 10;
    }
    result = *(_QWORD *)a1;
    v20 = *(_BYTE *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= v20;
    if ( result )
    {
      sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      return (unsigned __int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)v4);
    }
  }
  else
  {
    v14 = *(_QWORD *)a1;
    v23 = 1;
    v21[0] = "invalid tag";
    v22 = 3;
    if ( !v14 )
      goto LABEL_20;
    sub_16E2CE0(v21, v14);
    v15 = *(_BYTE **)(v14 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
    {
      sub_16E7DE0(v14, 10);
    }
    else
    {
      *(_QWORD *)(v14 + 24) = v15 + 1;
      *v15 = 10;
    }
    v16 = *(_QWORD *)a1;
    result = *(unsigned __int8 *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= result;
    if ( v16 )
    {
      sub_15562E0((unsigned __int8 *)a2, v16, a1 + 16, *(_QWORD *)(a1 + 8));
      v17 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
        return sub_16E7DE0(v17, 10);
      }
      else
      {
        *(_QWORD *)(v17 + 24) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  return result;
}
