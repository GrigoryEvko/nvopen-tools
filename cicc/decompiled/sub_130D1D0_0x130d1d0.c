// Function: sub_130D1D0
// Address: 0x130d1d0
//
__int64 __fastcall sub_130D1D0(__int64 a1, __int64 a2, __int64 *a3, _BYTE *a4)
{
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rax
  _BYTE v10[49]; // [rsp+Fh] [rbp-31h] BYREF

  v5 = *a3;
  if ( v5 )
  {
    do
    {
      v9 = *(_QWORD *)(v5 + 40);
      v10[0] = 0;
      *a3 = v9;
      if ( v9 == v5 )
      {
        *a3 = 0;
      }
      else
      {
        *(_QWORD *)(*(_QWORD *)(v5 + 48) + 40LL) = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL);
        v7 = *(_QWORD *)(v5 + 48);
        *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL) = v7;
        *(_QWORD *)(v5 + 48) = *(_QWORD *)(v7 + 40);
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL) + 40LL) = *(_QWORD *)(v5 + 40);
        *(_QWORD *)(*(_QWORD *)(v5 + 48) + 40LL) = v5;
      }
      (*(void (__fastcall **)(__int64, __int64, __int64, _BYTE *))(a2 + 32))(a1, a2, v5, v10);
      result = v10[0];
      *a4 |= v10[0];
      v5 = *a3;
    }
    while ( *a3 );
  }
  return result;
}
