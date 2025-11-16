// Function: sub_172C850
// Address: 0x172c850
//
__int64 __fastcall sub_172C850(__int64 a1, __int64 a2, double a3, double a4, double a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 v11; // rdi
  unsigned __int8 v13; // al
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+18h] [rbp-48h] BYREF
  __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-38h]
  __int16 v30; // [rsp+30h] [rbp-30h]

  v7 = *(_QWORD *)(a1 - 48);
  if ( *(_BYTE *)(v7 + 16) != 78 )
    return 0;
  v8 = *(_QWORD *)(v7 - 24);
  if ( *(_BYTE *)(v8 + 16) )
    return 0;
  if ( *(_DWORD *)(v8 + 36) != 6 )
    return 0;
  v9 = *(_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( !v9 )
    return 0;
  v11 = *(_QWORD *)(a1 - 24);
  v13 = *(_BYTE *)(v11 + 16);
  if ( v13 == 78 )
  {
    v22 = *(_QWORD *)(v11 - 24);
    if ( !*(_BYTE *)(v22 + 16) && *(_DWORD *)(v22 + 36) == 6 )
    {
      v17 = *(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL)
                      - 24LL * (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      if ( v17 )
      {
        v23 = *(_QWORD *)(v7 + 8);
        if ( v23 )
        {
          if ( !*(_QWORD *)(v23 + 8) )
            goto LABEL_13;
        }
        v24 = *(_QWORD *)(v11 + 8);
        if ( v24 )
        {
          if ( !*(_QWORD *)(v24 + 8) )
            goto LABEL_13;
        }
      }
    }
  }
  else
  {
    if ( v13 == 13 )
    {
      v14 = v11 + 24;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
        return 0;
      if ( v13 > 0x10u )
        return 0;
      v25 = sub_15A1020((_BYTE *)v11, a2, *(_QWORD *)v11, a7);
      if ( !v25 )
        return 0;
      v14 = v25 + 24;
      if ( *(_BYTE *)(v25 + 16) != 13 )
        return 0;
    }
    v15 = *(_QWORD *)(v7 + 8);
    if ( v15 && !*(_QWORD *)(v15 + 8) )
    {
      sub_16A85B0((__int64)&v28, v14);
      v16 = sub_15A1070(*(_QWORD *)a1, (__int64)&v28);
      v17 = v16;
      if ( v29 > 0x40 )
      {
        if ( v28 )
        {
          v26 = v16;
          j_j___libc_free_0_0(v28);
          v17 = v26;
        }
      }
LABEL_13:
      v18 = *(unsigned __int8 *)(a1 + 16) - 24;
      v30 = 257;
      v27 = sub_17066B0(a2, v18, v9, v17, &v28, 0, a3, a4, a5);
      v28 = *(_QWORD *)a1;
      v19 = (__int64 *)sub_15F2050(a1);
      v20 = sub_15E26F0(v19, 6, &v28, 1);
      v30 = 257;
      return sub_172C570(a2, *(_QWORD *)(v20 + 24), v20, &v27, 1, &v28, 0);
    }
  }
  return 0;
}
