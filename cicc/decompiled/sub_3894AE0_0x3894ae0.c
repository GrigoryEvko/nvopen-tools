// Function: sub_3894AE0
// Address: 0x3894ae0
//
__int64 __fastcall sub_3894AE0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rbx
  __int64 result; // rax
  __int64 v7; // r14
  _BYTE *v8; // rsi
  _QWORD v9[7]; // [rsp+8h] [rbp-38h] BYREF

  *(_QWORD *)(a1 + 40) = a1 + 24;
  *(_QWORD *)(a1 + 48) = a1 + 24;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 72;
  *(_QWORD *)(a1 + 96) = a1 + 72;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = a4;
  if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
  {
    sub_15E08E0(a3, a2);
    v5 = *(_QWORD *)(a3 + 88);
    result = 5LL * *(_QWORD *)(a3 + 96);
    v7 = v5 + 40LL * *(_QWORD *)(a3 + 96);
    if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
    {
      result = sub_15E08E0(a3, a2);
      v5 = *(_QWORD *)(a3 + 88);
    }
  }
  else
  {
    v5 = *(_QWORD *)(a3 + 88);
    result = 5LL * *(_QWORD *)(a3 + 96);
    v7 = v5 + 40LL * *(_QWORD *)(a3 + 96);
  }
  for ( ; v7 != v5; *(_QWORD *)(a1 + 120) = v8 + 8 )
  {
    while ( (*(_BYTE *)(v5 + 23) & 0x20) != 0 )
    {
LABEL_5:
      v5 += 40;
      if ( v7 == v5 )
        return result;
    }
    v9[0] = v5;
    v8 = *(_BYTE **)(a1 + 120);
    if ( v8 == *(_BYTE **)(a1 + 128) )
    {
      result = (__int64)sub_12879C0(a1 + 112, v8, v9);
      goto LABEL_5;
    }
    if ( v8 )
    {
      *(_QWORD *)v8 = v5;
      v8 = *(_BYTE **)(a1 + 120);
    }
    v5 += 40;
  }
  return result;
}
