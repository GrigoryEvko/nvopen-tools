// Function: sub_1655F00
// Address: 0x1655f00
//
unsigned __int64 __fastcall sub_1655F00(_BYTE *a1, __int64 a2, unsigned __int8 **a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  __int64 v7; // rsi
  unsigned __int64 result; // rax
  __int64 v9; // rdi

  v4 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    sub_16E2CE0(a2, v4);
    v6 = *(_BYTE **)(v4 + 24);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 16) )
    {
      sub_16E7DE0(v4, 10);
    }
    else
    {
      *(_QWORD *)(v4 + 24) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_QWORD *)a1;
    result = (unsigned __int8)a1[74];
    a1[73] = 1;
    a1[72] |= result;
    if ( v7 && *a3 )
    {
      sub_15562E0(*a3, v7, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
      v9 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
        return sub_16E7DE0(v9, 10);
      }
      else
      {
        *(_QWORD *)(v9 + 24) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  else
  {
    a1[73] = 1;
    result = (unsigned __int8)a1[74];
    a1[72] |= result;
  }
  return result;
}
