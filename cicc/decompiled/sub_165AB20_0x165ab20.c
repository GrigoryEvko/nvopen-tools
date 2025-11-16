// Function: sub_165AB20
// Address: 0x165ab20
//
unsigned __int64 __fastcall sub_165AB20(__int64 *a1, __int64 a2, __int64 *a3, unsigned __int8 **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 result; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi

  v4 = *a1;
  if ( !*a1 )
  {
    *((_BYTE *)a1 + 73) = 1;
    result = *((unsigned __int8 *)a1 + 74);
    *((_BYTE *)a1 + 72) |= result;
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
  v9 = *a1;
  result = *((unsigned __int8 *)a1 + 74);
  *((_BYTE *)a1 + 73) = 1;
  *((_BYTE *)a1 + 72) |= result;
  if ( v9 )
  {
    v11 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *(_BYTE *)(v11 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)v11, v9, 1, (__int64)(a1 + 2));
      v12 = *a1;
      result = *(_QWORD *)(*a1 + 24);
      if ( result < *(_QWORD *)(*a1 + 16) )
        goto LABEL_8;
    }
    else
    {
      sub_155BD40(v11, v9, (__int64)(a1 + 2), 0);
      v12 = *a1;
      result = *(_QWORD *)(*a1 + 24);
      if ( result < *(_QWORD *)(*a1 + 16) )
      {
LABEL_8:
        *(_QWORD *)(v12 + 24) = result + 1;
        *(_BYTE *)result = 10;
        goto LABEL_9;
      }
    }
    result = sub_16E7DE0(v12, 10);
LABEL_9:
    if ( *a4 )
    {
      sub_15562E0(*a4, *a1, (__int64)(a1 + 2), a1[1]);
      v13 = *a1;
      result = *(_QWORD *)(*a1 + 24);
      if ( result >= *(_QWORD *)(*a1 + 16) )
      {
        return sub_16E7DE0(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 24) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  return result;
}
