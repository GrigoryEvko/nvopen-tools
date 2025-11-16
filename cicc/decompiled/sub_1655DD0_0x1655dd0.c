// Function: sub_1655DD0
// Address: 0x1655dd0
//
unsigned __int64 __fastcall sub_1655DD0(__int64 a1, __int64 a2, unsigned __int8 **a3, unsigned __int8 **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 result; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi

  v4 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
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
      if ( *a4 )
      {
        sub_15562E0(*a4, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
        v12 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          return sub_16E7DE0(v12, 10);
        }
        else
        {
          *(_QWORD *)(v12 + 24) = result + 1;
          *(_BYTE *)result = 10;
        }
      }
    }
  }
  else
  {
    *(_BYTE *)(a1 + 73) = 1;
    result = *(unsigned __int8 *)(a1 + 74);
    *(_BYTE *)(a1 + 72) |= result;
  }
  return result;
}
