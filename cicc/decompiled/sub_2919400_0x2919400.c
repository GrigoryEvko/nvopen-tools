// Function: sub_2919400
// Address: 0x2919400
//
char __fastcall sub_2919400(__int64 *a1, __int64 *a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // r8
  int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r14
  unsigned __int8 v25; // dl
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rdi
  unsigned int v30; // r12d
  int v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]

  v10 = *a1;
  v11 = *a2;
  if ( *a1 > (unsigned __int64)*a2 )
  {
    v12 = 0;
  }
  else
  {
    v12 = (v11 - v10) / a4;
    if ( v12 * a4 != v11 - v10 )
    {
LABEL_3:
      LOBYTE(v13) = 0;
      return v13;
    }
  }
  v14 = *(unsigned int *)(a3 + 32);
  if ( v14 <= v12 )
    goto LABEL_3;
  v15 = a2[1];
  if ( a1[1] <= v15 )
    v15 = a1[1];
  v16 = v15 - v10;
  v17 = v16 / a4;
  if ( v16 / a4 * a4 != v16 || v14 < v17 )
    goto LABEL_3;
  v18 = v17 - v12;
  v19 = *(_QWORD *)(a3 + 24);
  v20 = v18;
  if ( v18 != 1 )
  {
    v32 = v18;
    v21 = sub_BCDA70(*(__int64 **)(a3 + 24), v18);
    v20 = v32;
    v19 = v21;
  }
  v33 = v19;
  v22 = sub_BCD140(*(_QWORD **)a3, 8 * (int)a4 * v20);
  v23 = a2[2];
  v24 = *(_QWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v25 = *(_BYTE *)v24;
  if ( *(_BYTE *)v24 == 85 )
  {
    v28 = *(_QWORD *)(v24 - 32);
    if ( !v28 )
      goto LABEL_3;
    if ( !*(_BYTE *)v28
      && *(_QWORD *)(v28 + 24) == *(_QWORD *)(v24 + 80)
      && (*(_BYTE *)(v28 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v28 + 36) - 238) <= 7
      && ((1LL << (*(_BYTE *)(v28 + 36) + 18)) & 0xAD) != 0 )
    {
      v29 = *(_QWORD *)(v24 + 32 * (3LL - (*(_DWORD *)(v24 + 4) & 0x7FFFFFF)));
      v30 = *(_DWORD *)(v29 + 32);
      if ( v30 <= 0x40 )
      {
        if ( *(_QWORD *)(v29 + 24) )
          goto LABEL_3;
      }
      else if ( v30 != (unsigned int)sub_C444A0(v29 + 24) )
      {
        goto LABEL_3;
      }
      return (v23 >> 2) & 1;
    }
    else
    {
      if ( *(_BYTE *)v28 || *(_QWORD *)(v28 + 24) != *(_QWORD *)(v24 + 80) || (*(_BYTE *)(v28 + 33) & 0x20) == 0 )
        goto LABEL_3;
      LOBYTE(v13) = sub_B46A10(*(_QWORD *)((a2[2] & 0xFFFFFFFFFFFFFFF8LL) + 24));
      if ( !(_BYTE)v13 )
        LOBYTE(v13) = sub_BD2BE0(v24);
    }
  }
  else
  {
    if ( v25 <= 0x1Cu )
      goto LABEL_3;
    if ( v25 == 61 )
    {
      if ( (*(_BYTE *)(v24 + 2) & 1) != 0 || *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL) == 15 )
        goto LABEL_3;
      if ( *a2 >= (unsigned __int64)*a1 && a2[1] <= (unsigned __int64)a1[1] )
        v22 = *(_QWORD *)(v24 + 8);
      v26 = v22;
      v27 = v33;
    }
    else
    {
      if ( v25 != 62
        || (*(_BYTE *)(v24 + 2) & 1) != 0
        || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v24 - 64) + 8LL) + 8LL) == 15 )
      {
        goto LABEL_3;
      }
      if ( *a2 >= (unsigned __int64)*a1 && a2[1] <= (unsigned __int64)a1[1] )
        v22 = *(_QWORD *)(*(_QWORD *)(v24 - 64) + 8LL);
      v26 = v33;
      v27 = v22;
    }
    LOBYTE(v13) = sub_29191E0(a5, v27, v26);
  }
  return v13;
}
