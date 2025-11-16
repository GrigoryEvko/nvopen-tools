// Function: sub_16C1EA0
// Address: 0x16c1ea0
//
__int64 __fastcall sub_16C1EA0(__int64 a1, __int64 (*a2)(void), __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r14
  __int64 v11; // rax
  int v12; // r14d

  if ( (unsigned __int8)sub_16D5D40(a1, a2, a3, a4, a5, a6) )
  {
    if ( (unsigned int)sub_16AF4C0(&dword_4FA04D0, 1, 0) )
    {
      do
      {
        v12 = dword_4FA04D0;
        sub_16AF4B0();
      }
      while ( v12 != 2 );
    }
    else
    {
      v8 = sub_22077B0(16);
      v9 = v8;
      if ( v8 )
      {
        sub_16C3010(v8, 1);
        *(_DWORD *)(v9 + 8) = 0;
        *(_BYTE *)(v9 + 12) = 1;
      }
      qword_4FA04D8 = v9;
      sub_16AF4B0();
      dword_4FA04D0 = 2;
    }
    v10 = qword_4FA04D8;
    sub_16C30C0(qword_4FA04D8);
    if ( !*(_QWORD *)a1 )
    {
      *(_QWORD *)a1 = a2();
      *(_QWORD *)(a1 + 8) = a3;
      v11 = qword_4FA04E0;
      qword_4FA04E0 = a1;
      *(_QWORD *)(a1 + 16) = v11;
    }
    return sub_16C30E0(v10);
  }
  else
  {
    _InterlockedExchange64((volatile __int64 *)a1, a2());
    result = qword_4FA04E0;
    *(_QWORD *)(a1 + 8) = a3;
    *(_QWORD *)(a1 + 16) = result;
    qword_4FA04E0 = a1;
  }
  return result;
}
