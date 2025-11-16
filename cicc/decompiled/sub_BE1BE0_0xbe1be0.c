// Function: sub_BE1BE0
// Address: 0xbe1be0
//
void __fastcall sub_BE1BE0(_BYTE *a1, __int64 a2, const char **a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  _BYTE *v9; // rax

  v4 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    sub_CA0E80(a2, v4);
    v6 = *(_BYTE **)(v4 + 32);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 24) )
    {
      sub_CB5D20(v4, 10);
    }
    else
    {
      *(_QWORD *)(v4 + 32) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_QWORD *)a1;
    a1[152] = 1;
    if ( v7 && *a3 )
    {
      sub_A62C00(*a3, v7, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
      v8 = *(_QWORD *)a1;
      v9 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v9 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        sub_CB5D20(v8, 10);
      }
      else
      {
        *(_QWORD *)(v8 + 32) = v9 + 1;
        *v9 = 10;
      }
    }
  }
  else
  {
    a1[152] = 1;
  }
}
