// Function: sub_165AC70
// Address: 0x165ac70
//
unsigned __int64 __fastcall sub_165AC70(__int64 a1, __int64 a2, unsigned __int8 **a3, __int64 *a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 result; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdi

  v4 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    *(_BYTE *)(a1 + 73) = 1;
    result = *(unsigned __int8 *)(a1 + 74);
    *(_BYTE *)(a1 + 72) |= result;
    return result;
  }
  sub_16E2CE0(a2, v4);
  v8 = *(_BYTE **)(v4 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v4 + 16) )
  {
    sub_16E7DE0(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v8 + 1;
    *v8 = 10;
  }
  v9 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(a1 + 74);
  *(_BYTE *)(a1 + 73) = 1;
  *(_BYTE *)(a1 + 72) |= result;
  if ( v9 )
  {
    if ( *a3 )
    {
      sub_15562E0(*a3, v9, a1 + 16, *(_QWORD *)(a1 + 8));
      v11 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
        result = sub_16E7DE0(v11, 10);
      }
      else
      {
        *(_QWORD *)(v11 + 24) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
    v12 = *a4;
    if ( *a4 )
    {
      v13 = *(_QWORD *)a1;
      if ( *(_BYTE *)(v12 + 16) <= 0x17u )
      {
        sub_1553920((__int64 *)v12, v13, 1, a1 + 16);
        v14 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( result < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          goto LABEL_11;
      }
      else
      {
        sub_155BD40(v12, v13, a1 + 16, 0);
        v14 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( result < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
LABEL_11:
          *(_QWORD *)(v14 + 24) = result + 1;
          *(_BYTE *)result = 10;
          return result;
        }
      }
      return sub_16E7DE0(v14, 10);
    }
  }
  return result;
}
