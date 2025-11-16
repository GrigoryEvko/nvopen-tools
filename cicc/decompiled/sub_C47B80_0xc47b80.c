// Function: sub_C47B80
// Address: 0xc47b80
//
__int64 __fastcall sub_C47B80(__int64 a1, __int64 a2, unsigned int a3, bool *a4)
{
  bool v4; // cf
  unsigned int v5; // eax
  unsigned int v7; // eax
  unsigned int v10; // ecx
  unsigned __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx

  v4 = a3 < *(_DWORD *)(a2 + 8);
  *a4 = a3 >= *(_DWORD *)(a2 + 8);
  if ( v4 )
  {
    v7 = *(_DWORD *)(a2 + 8);
    if ( v7 > 0x40 )
    {
      v7 = sub_C444A0(a2);
    }
    else
    {
      v10 = v7 - 64;
      if ( *(_QWORD *)a2 )
      {
        _BitScanReverse64(&v11, *(_QWORD *)a2);
        v7 = v10 + (v11 ^ 0x3F);
      }
    }
    *a4 = a3 > v7;
    v12 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v12;
    if ( v12 > 0x40 )
    {
      sub_C43780(a1, (const void **)a2);
      v12 = *(_DWORD *)(a1 + 8);
      if ( v12 > 0x40 )
      {
        sub_C47690((__int64 *)a1, a3);
        return a1;
      }
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a2;
    }
    v13 = 0;
    if ( a3 != v12 )
      v13 = *(_QWORD *)a1 << a3;
    v14 = v13 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
    if ( !v12 )
      v14 = 0;
    *(_QWORD *)a1 = v14;
    return a1;
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v5;
    if ( v5 > 0x40 )
      sub_C43690(a1, 0, 0);
    else
      *(_QWORD *)a1 = 0;
    return a1;
  }
}
