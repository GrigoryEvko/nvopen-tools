// Function: sub_C4B490
// Address: 0xc4b490
//
__int64 __fastcall sub_C4B490(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r9d
  int v5; // eax
  unsigned int v6; // r9d
  unsigned int v7; // r15d
  unsigned __int64 v8; // r14
  int v9; // eax
  unsigned int v10; // r15d
  unsigned __int64 v11; // rcx
  unsigned int v12; // r15d
  int v13; // eax
  bool v14; // al
  unsigned __int64 v15; // rax
  __int64 v17; // rdx
  unsigned int v18; // [rsp+8h] [rbp-48h]
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  unsigned int v20; // [rsp+Ch] [rbp-44h]
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  if ( v4 <= 0x40 )
  {
    v17 = *(_QWORD *)a2 % *(_QWORD *)a3;
    *(_DWORD *)(a1 + 8) = v4;
    *(_QWORD *)a1 = v17;
    return a1;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v5 = sub_C444A0(a2);
  v6 = v19;
  v7 = *(_DWORD *)(a3 + 8);
  v8 = ((unsigned __int64)(v19 - v5) + 63) >> 6;
  if ( v7 <= 0x40 )
  {
    if ( !*(_QWORD *)a3 )
    {
      v12 = ((unsigned __int64)(v19 - v5) + 63) >> 6;
      LODWORD(v11) = 0;
      if ( !v8 )
        goto LABEL_12;
      goto LABEL_6;
    }
    _BitScanReverse64(&v15, *(_QWORD *)a3);
    LODWORD(v11) = 1;
    v10 = 64 - (v15 ^ 0x3F);
    if ( !v8 )
      goto LABEL_12;
  }
  else
  {
    v9 = sub_C444A0(a3);
    v6 = v19;
    v10 = v7 - v9;
    v11 = ((unsigned __int64)v10 + 63) >> 6;
    if ( !v8 )
    {
LABEL_12:
      *(_DWORD *)(a1 + 8) = v6;
      sub_C43690(a1, 0, 0);
      return a1;
    }
  }
  if ( v10 == 1 )
    goto LABEL_12;
  v12 = v8;
  if ( (unsigned int)v8 < (unsigned int)v11 )
  {
LABEL_13:
    *(_DWORD *)(a1 + 8) = v6;
    sub_C43780(a1, (const void **)a2);
    return a1;
  }
LABEL_6:
  v18 = v11;
  v20 = v6;
  v13 = sub_C49970(a2, (unsigned __int64 *)a3);
  v6 = v20;
  if ( v13 < 0 )
    goto LABEL_13;
  v14 = sub_C43C50(a2, (const void **)a3);
  v6 = v20;
  if ( v14 )
    goto LABEL_12;
  if ( v12 == 1 )
  {
    sub_9691E0(a1, v20, **(_QWORD **)a2 % **(_QWORD **)a3, 0, 0);
  }
  else
  {
    v22 = v20;
    sub_C43690((__int64)&v21, 0, 0);
    sub_C44DF0(*(__int64 **)a2, v12, *(__int64 **)a3, v18, 0, v21);
    *(_DWORD *)(a1 + 8) = v22;
    *(_QWORD *)a1 = v21;
  }
  return a1;
}
