// Function: sub_15A0D80
// Address: 0x15a0d80
//
__int64 __fastcall sub_15A0D80(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  _BYTE *v6; // rdi
  unsigned int v8; // r13d
  int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdi

  if ( a1[16] == 14 )
  {
    v5 = sub_16982C0(a1, a2, a3, a4);
    v6 = a1 + 32;
    if ( *((_QWORD *)a1 + 4) == v5 )
      return sub_16A25B0(v6, 0);
    else
      return sub_16A0030(v6, 0);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    {
      v8 = 0;
      v9 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v9 )
        return 1;
      while ( 1 )
      {
        v10 = sub_15A0A60((__int64)a1, v8);
        if ( !v10 || *(_BYTE *)(v10 + 16) != 14 )
          break;
        v13 = v10 + 32;
        if ( !(*(_QWORD *)(v10 + 32) == sub_16982C0(a1, v8, v11, v12)
             ? sub_16A25B0(v13, 0)
             : (unsigned __int8)sub_16A0030(v13, 0)) )
          break;
        if ( v9 == ++v8 )
          return 1;
      }
    }
    return 0;
  }
}
