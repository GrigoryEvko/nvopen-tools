// Function: sub_16589C0
// Address: 0x16589c0
//
void __fastcall sub_16589C0(_BYTE *a1, __int64 a2, unsigned __int8 **a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  _BYTE *v9; // rax

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
    a1[72] = 1;
    if ( v7 && *a3 )
    {
      sub_15562E0(*a3, v7, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
      v8 = *(_QWORD *)a1;
      v9 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v9 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
        sub_16E7DE0(v8, 10);
      }
      else
      {
        *(_QWORD *)(v8 + 24) = v9 + 1;
        *v9 = 10;
      }
    }
  }
  else
  {
    a1[72] = 1;
  }
}
