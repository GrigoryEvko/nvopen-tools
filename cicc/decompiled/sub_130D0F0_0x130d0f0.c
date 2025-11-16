// Function: sub_130D0F0
// Address: 0x130d0f0
//
__int64 __fastcall sub_130D0F0(
        __int64 a1,
        __int64 (__fastcall **a2)(__int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _BYTE *),
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _BYTE *a6)
{
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v12; // [rsp+18h] [rbp-48h]
  _BYTE v13[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v12 = a4;
  if ( a4 )
  {
    v9 = 0;
    while ( 1 )
    {
      v13[0] = 0;
      v10 = (*a2)(a1, a2, a3, 4096, 0, 0, 0, v13);
      *a6 |= v13[0];
      if ( !v10 )
        break;
      *(_QWORD *)(v10 + 40) = v10;
      *(_QWORD *)(v10 + 48) = v10;
      if ( *(_QWORD *)a5 )
      {
        *(_QWORD *)(v10 + 40) = *(_QWORD *)(*(_QWORD *)a5 + 48LL);
        *(_QWORD *)(*(_QWORD *)a5 + 48LL) = v10;
        *(_QWORD *)(v10 + 48) = *(_QWORD *)(*(_QWORD *)(v10 + 48) + 40LL);
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a5 + 48LL) + 40LL) = *(_QWORD *)a5;
        *(_QWORD *)(*(_QWORD *)(v10 + 48) + 40LL) = v10;
        v10 = *(_QWORD *)(v10 + 40);
      }
      *(_QWORD *)a5 = v10;
      if ( v12 == ++v9 )
        return v12;
    }
    return v9;
  }
  return v12;
}
