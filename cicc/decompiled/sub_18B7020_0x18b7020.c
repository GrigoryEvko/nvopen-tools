// Function: sub_18B7020
// Address: 0x18b7020
//
__int64 __fastcall sub_18B7020(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 i; // r14
  __int64 *v5; // rax
  __int64 v6; // rsi
  unsigned __int64 v7; // rcx
  __int64 v8; // rcx
  _DWORD *v9; // rax
  unsigned __int64 v10; // rax
  __int64 ****v11; // rcx
  __int64 ****v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 result; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  for ( i = *(_QWORD *)a2; v3 != i; i += 24 )
  {
    if ( *(_BYTE *)(*a1 + 80) )
    {
      v19 = *a1;
      v15 = (char *)sub_1649960(*(_QWORD *)a1[1]);
      sub_18B6C20(
        i,
        "single-impl",
        11,
        v15,
        v16,
        v17,
        *(__int64 (__fastcall **)(__int64, __int64))(v19 + 88),
        *(_QWORD *)(v19 + 96));
    }
    v10 = *(_QWORD *)(i + 8) & 0xFFFFFFFFFFFFFFF8LL;
    v11 = (__int64 ****)(v10 - 24);
    v12 = (__int64 ****)(v10 - 72);
    if ( (*(_QWORD *)(i + 8) & 4) != 0 )
      v12 = v11;
    v13 = sub_15A4510(*(__int64 ****)a1[1], **v12, 0);
    v14 = *(_QWORD *)(i + 8);
    if ( (v14 & 4) != 0 )
      v5 = (__int64 *)((v14 & 0xFFFFFFFFFFFFFFF8LL) - 24);
    else
      v5 = (__int64 *)((v14 & 0xFFFFFFFFFFFFFFF8LL) - 72);
    if ( *v5 )
    {
      v6 = v5[1];
      v7 = v5[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v7 = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
    }
    *v5 = v13;
    if ( v13 )
    {
      v8 = *(_QWORD *)(v13 + 8);
      v5[1] = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = (unsigned __int64)(v5 + 1) | *(_QWORD *)(v8 + 16) & 3LL;
      v5[2] = (v13 + 8) | v5[2] & 3;
      *(_QWORD *)(v13 + 8) = v5;
    }
    v9 = *(_DWORD **)(i + 16);
    if ( v9 )
      --*v9;
  }
  if ( !*(_BYTE *)(a2 + 25) && (result = *(_QWORD *)(a2 + 32), *(_QWORD *)(a2 + 40) == result) )
  {
    *(_BYTE *)(a2 + 24) = 1;
  }
  else
  {
    *(_BYTE *)a1[2] = 1;
    result = *(_QWORD *)(a2 + 32);
    *(_BYTE *)(a2 + 24) = 1;
    if ( result != *(_QWORD *)(a2 + 40) )
      *(_QWORD *)(a2 + 40) = result;
  }
  return result;
}
