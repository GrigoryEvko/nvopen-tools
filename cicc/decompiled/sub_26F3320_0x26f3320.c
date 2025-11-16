// Function: sub_26F3320
// Address: 0x26f3320
//
char __fastcall sub_26F3320(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v22; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v23[80]; // [rsp+20h] [rbp-50h] BYREF

  if ( *(_BYTE *)a1 )
  {
    LOBYTE(v6) = *(_BYTE *)a1 - 1;
    if ( (unsigned __int8)v6 > 2u )
    {
      v6 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v7 = *(__int64 **)(a1 - 8);
        v8 = (__int64 *)((char *)v7 + v6);
      }
      else
      {
        v8 = (__int64 *)a1;
        v7 = (__int64 *)(a1 - v6);
      }
      for ( ; v8 != v7; LOBYTE(v6) = sub_26F3320(v9, a2, a3) )
      {
        v9 = *v7;
        v7 += 4;
      }
    }
    return v6;
  }
  v5 = *(_QWORD *)(a1 + 24);
  v22 = a1;
  v6 = **(_QWORD **)(v5 + 16);
  if ( *(_BYTE *)(v6 + 8) != 12 || *(_DWORD *)(v6 + 8) > 0x40FFu || !*(_QWORD *)(a1 + 104) )
    return v6;
  if ( (*(_BYTE *)(a1 + 2) & 1) == 0 )
  {
    v6 = *(_QWORD *)(a1 + 96);
    if ( *(_QWORD *)(v6 + 16) )
      return v6;
    v12 = a1;
    goto LABEL_20;
  }
  sub_B2C6D0(a1, a2, (__int64)a3, a4);
  v6 = *(_QWORD *)(a1 + 96);
  if ( *(_QWORD *)(v6 + 16) )
    return v6;
  v12 = v22;
  if ( (*(_BYTE *)(v22 + 2) & 1) == 0 )
  {
LABEL_20:
    v15 = *(_QWORD *)(v12 + 96);
    v16 = v15 + 40LL * *(_QWORD *)(v12 + 104);
    goto LABEL_21;
  }
  sub_B2C6D0(v22, a2, v10, v11);
  v15 = *(_QWORD *)(v22 + 96);
  v16 = v15 + 40LL * *(_QWORD *)(v22 + 104);
  if ( (*(_BYTE *)(v22 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v22, a2, v13, v14);
    v15 = *(_QWORD *)(v22 + 96);
  }
LABEL_21:
  v6 = sub_26F15C0(v15, v16, 1);
  v18 = v17;
  if ( v6 == v17 )
  {
LABEL_25:
    LOBYTE(v6) = sub_B2FC80(v22);
    if ( !(_BYTE)v6 )
    {
      v20 = (*(__int64 (__fastcall **)(_QWORD, __int64))*a3)(*(_QWORD *)(*a3 + 8), v22);
      LODWORD(v6) = sub_25BFB50(v22, v20);
      if ( !(_DWORD)v6 )
        LOBYTE(v6) = sub_2686B10((__int64)v23, a3[1], &v22);
    }
  }
  else
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v6 + 8);
      if ( *(_BYTE *)(v19 + 8) != 12 || *(_DWORD *)(v19 + 8) > 0x40FFu )
        break;
      v6 += 40;
      if ( v18 == v6 )
        goto LABEL_25;
    }
  }
  return v6;
}
