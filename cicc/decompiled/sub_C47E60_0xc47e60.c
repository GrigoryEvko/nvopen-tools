// Function: sub_C47E60
// Address: 0xc47e60
//
__int64 __fastcall sub_C47E60(__int64 a1, __int64 a2, unsigned int a3, bool *a4)
{
  bool v5; // cf
  unsigned int v7; // edx
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  unsigned int v15; // eax
  unsigned __int64 v16; // rax

  v5 = a3 < *(_DWORD *)(a2 + 8);
  *a4 = a3 >= *(_DWORD *)(a2 + 8);
  if ( !v5 )
  {
    v15 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v15;
    if ( v15 > 0x40 )
      sub_C43690(a1, 0, 0);
    else
      *(_QWORD *)a1 = 0;
    return a1;
  }
  v7 = *(_DWORD *)(a2 + 8);
  v9 = 1LL << ((unsigned __int8)v7 - 1);
  v10 = *(_QWORD *)a2;
  if ( v7 > 0x40 )
  {
    if ( (*(_QWORD *)(v10 + 8LL * ((v7 - 1) >> 6)) & v9) == 0 )
    {
      v7 = sub_C444A0(a2);
      goto LABEL_6;
    }
    *a4 = a3 >= (unsigned int)sub_C44500(a2);
LABEL_21:
    v11 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v11;
    if ( v11 <= 0x40 )
      goto LABEL_7;
LABEL_22:
    sub_C43780(a1, (const void **)a2);
    v11 = *(_DWORD *)(a1 + 8);
    if ( v11 > 0x40 )
    {
      sub_C47690((__int64 *)a1, a3);
      return a1;
    }
    goto LABEL_8;
  }
  if ( (v9 & v10) != 0 )
  {
    if ( v7 )
    {
      v16 = ~(v10 << (64 - (unsigned __int8)v7));
      if ( v16 )
      {
        _BitScanReverse64(&v16, v16);
        *a4 = a3 >= ((unsigned int)v16 ^ 0x3F);
      }
      else
      {
        *a4 = a3 > 0x3F;
      }
    }
    else
    {
      *a4 = 1;
    }
    goto LABEL_21;
  }
  if ( v10 )
  {
    _BitScanReverse64(&v10, v10);
    v7 = v7 - 64 + (v10 ^ 0x3F);
  }
LABEL_6:
  *a4 = a3 >= v7;
  v11 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v11;
  if ( v11 > 0x40 )
    goto LABEL_22;
LABEL_7:
  *(_QWORD *)a1 = *(_QWORD *)a2;
LABEL_8:
  v12 = 0;
  if ( a3 != v11 )
    v12 = *(_QWORD *)a1 << a3;
  v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & v12;
  if ( !v11 )
    v13 = 0;
  *(_QWORD *)a1 = v13;
  return a1;
}
