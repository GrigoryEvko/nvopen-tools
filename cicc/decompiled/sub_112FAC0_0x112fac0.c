// Function: sub_112FAC0
// Address: 0x112fac0
//
__int64 __fastcall sub_112FAC0(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v4; // rax
  _BYTE *v5; // rax
  unsigned int v6; // r13d
  __int64 v7; // r14
  __int64 v8; // rax
  int v10; // r13d
  int v11; // r12d

  if ( *(_BYTE *)a2 == 17 )
  {
    v3 = *(_DWORD *)(a2 + 32);
    if ( v3 > 0x40 )
    {
      v10 = sub_C445E0(a2 + 24);
      if ( v10 && v3 == (unsigned int)sub_C444A0(a2 + 24) + v10 )
        goto LABEL_17;
    }
    else
    {
      v4 = *(_QWORD *)(a2 + 24);
      if ( v4 )
      {
        a3 = v4 + 1;
        if ( (v4 & (v4 + 1)) == 0 )
        {
LABEL_17:
          **a1 = a2 + 24;
          return 1;
        }
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1 )
      return 0;
  }
  else
  {
    a3 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)a3 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
  }
  v5 = sub_AD7630(a2, 1, a3);
  if ( !v5 || *v5 != 17 )
    return 0;
  v6 = *((_DWORD *)v5 + 8);
  v7 = (__int64)(v5 + 24);
  if ( v6 > 0x40 )
  {
    v11 = sub_C445E0((__int64)(v5 + 24));
    if ( !v11 || v6 != (unsigned int)sub_C444A0(v7) + v11 )
      return 0;
  }
  else
  {
    v8 = *((_QWORD *)v5 + 3);
    if ( !v8 || (v8 & (v8 + 1)) != 0 )
      return 0;
  }
  **a1 = v7;
  return 1;
}
