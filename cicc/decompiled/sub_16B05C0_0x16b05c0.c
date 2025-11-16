// Function: sub_16B05C0
// Address: 0x16b05c0
//
__int64 __fastcall sub_16B05C0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 *a3,
        unsigned __int8 (__fastcall *a4)(_QWORD),
        __int64 a5)
{
  unsigned __int64 v6; // r14
  int v8; // eax
  __int64 v9; // r13

  v6 = a2;
  v8 = sub_16D1B30(a5, a1, a2);
  if ( v8 != -1 )
    goto LABEL_4;
  while ( v6 > 1 )
  {
    v8 = sub_16D1B30(a5, a1, --v6);
    if ( v8 != -1 )
    {
LABEL_4:
      v9 = *(_QWORD *)a5 + 8LL * v8;
      if ( v9 != *(_QWORD *)a5 + 8LL * *(unsigned int *)(a5 + 8) )
      {
        if ( a4(*(_QWORD *)(*(_QWORD *)v9 + 8LL)) )
        {
          *a3 = v6;
          return *(_QWORD *)(*(_QWORD *)v9 + 8LL);
        }
        return 0;
      }
    }
  }
  return 0;
}
