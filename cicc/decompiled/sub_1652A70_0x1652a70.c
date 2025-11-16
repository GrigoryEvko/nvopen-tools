// Function: sub_1652A70
// Address: 0x1652a70
//
unsigned __int64 __fastcall sub_1652A70(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // r14
  unsigned __int64 result; // rax
  __int64 v6; // r15
  _BYTE *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdi
  _QWORD v11[2]; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+10h] [rbp-30h]
  char v13; // [rsp+11h] [rbp-2Fh]

  v4 = *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)));
  if ( !v4 )
  {
LABEL_5:
    v6 = *(_QWORD *)a1;
    v13 = 1;
    v11[0] = "invalid local scope";
    v12 = 3;
    if ( v6 )
    {
      sub_16E2CE0(v11, v6);
      v7 = *(_BYTE **)(v6 + 24);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
      {
        sub_16E7DE0(v6, 10);
      }
      else
      {
        *(_QWORD *)(v6 + 24) = v7 + 1;
        *v7 = 10;
      }
      v8 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= result;
      if ( v8 )
      {
        sub_15562E0((unsigned __int8 *)a2, v8, a1 + 16, *(_QWORD *)(a1 + 8));
        v9 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          result = sub_16E7DE0(v9, 10);
        }
        else
        {
          *(_QWORD *)(v9 + 24) = result + 1;
          *(_BYTE *)result = 10;
        }
        if ( v4 )
        {
          sub_15562E0(v4, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
          v10 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
            return sub_16E7DE0(v10, 10);
          }
          else
          {
            *(_QWORD *)(v10 + 24) = result + 1;
            *(_BYTE *)result = 10;
          }
        }
      }
    }
    else
    {
      result = *(unsigned __int8 *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= result;
    }
    return result;
  }
  result = *v4;
  if ( (_DWORD)result != 17 )
  {
    result = (unsigned int)(result - 18);
    if ( (unsigned int)result <= 1 )
      return result;
    goto LABEL_5;
  }
  if ( (v4[40] & 8) == 0 )
  {
    v13 = 1;
    v11[0] = "scope points into the type hierarchy";
    v12 = 3;
    result = sub_16521E0((__int64 *)a1, (__int64)v11);
    if ( *(_QWORD *)a1 )
      return (unsigned __int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
  }
  return result;
}
