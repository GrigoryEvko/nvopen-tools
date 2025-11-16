// Function: sub_2ED4900
// Address: 0x2ed4900
//
__int64 __fastcall sub_2ED4900(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r8
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  __int64 result; // rax
  __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rcx
  __int64 v23; // rdx
  int *v24; // rcx
  int *v25; // r14
  int *v26; // r12
  __int64 v27; // rax
  int v28; // r10d
  __int64 v29; // rsi
  __int64 v30; // rdx
  signed __int64 v31; // rdx
  __int64 v32; // rdi
  __int64 v33; // rsi
  _DWORD *v34; // rax
  _DWORD *v35; // r12
  _DWORD *i; // r15
  __int64 *v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+28h] [rbp-48h]
  __int64 *v41; // [rsp+28h] [rbp-48h]
  __int64 v42; // [rsp+28h] [rbp-48h]
  _QWORD v43[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a2 + 48 == (*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)(a2 + 48) == a3 )
  {
    v33 = *(_QWORD *)(a1 + 56);
    v43[0] = 0;
    if ( v33 )
    {
      sub_B91220(a1 + 56, v33);
      *(_QWORD *)(a1 + 56) = v43[0];
    }
  }
  else
  {
    v40 = sub_B10CD0((__int64)(a3 + 7));
    v9 = sub_B10CD0(a1 + 56);
    v10 = sub_B026B0(v9, v40);
    sub_B10CB0(v43, (__int64)v10);
    v11 = *(_QWORD *)(a1 + 56);
    v12 = a1 + 56;
    if ( v11 )
    {
      sub_B91220(a1 + 56, v11);
      v12 = a1 + 56;
    }
    v13 = (unsigned __int8 *)v43[0];
    *(_QWORD *)(a1 + 56) = v43[0];
    if ( v13 )
      sub_B976B0((__int64)v43, v13, v12);
  }
  v14 = a1;
  if ( (*(_BYTE *)a1 & 4) == 0 && (*(_BYTE *)(a1 + 44) & 8) != 0 )
  {
    do
      v14 = *(_QWORD *)(v14 + 8);
    while ( (*(_BYTE *)(v14 + 44) & 8) != 0 );
  }
  v15 = *(_QWORD *)(v14 + 8);
  if ( a1 != v15 && a3 != (__int64 *)v15 )
  {
    v41 = (__int64 *)v15;
    sub_2E310C0((__int64 *)(a2 + 40), (__int64 *)(*(_QWORD *)(a1 + 24) + 40LL), a1, v15);
    if ( v41 != a3 && v41 != (__int64 *)a1 )
    {
      v16 = *v41 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v41;
      *v41 = *v41 & 7 | *(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL;
      v17 = *a3;
      *(_QWORD *)(v16 + 8) = a3;
      *(_QWORD *)a1 = v17 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)a1 & 7LL;
      *(_QWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 8) = a1;
      *a3 = v16 | *a3 & 7;
    }
  }
  v42 = a4;
  result = a4 + 32 * a5;
  v37 = (__int64 *)(a2 + 40);
  v38 = result;
  if ( a4 == result )
    return result;
  do
  {
    v19 = *(_QWORD *)v42;
    v20 = (_QWORD *)sub_2E88D60(*(_QWORD *)v42);
    v21 = (__int64)sub_2E7B2C0(v20, v19);
    sub_2E31040(v37, v21);
    v22 = *a3;
    v23 = *(_QWORD *)v21 & 7LL;
    *(_QWORD *)(v21 + 8) = a3;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v21 = v22 | v23;
    *(_QWORD *)(v22 + 8) = v21;
    *a3 = *a3 & 7 | v21;
    v24 = *(int **)(v42 + 8);
    v25 = &v24[*(unsigned int *)(v42 + 16)];
    if ( v25 == v24 )
      goto LABEL_42;
    v26 = *(int **)(v42 + 8);
    while ( 1 )
    {
      v27 = *(_QWORD *)(v19 + 32);
      v28 = *v26;
      v29 = v27 + 40;
      if ( *(_WORD *)(v19 + 68) == 14 )
        break;
      v30 = 40LL * (*(_DWORD *)(v19 + 40) & 0xFFFFFF);
      v29 = v27 + v30;
      v27 += 80;
      v31 = 0xCCCCCCCCCCCCCCCDLL * ((v30 - 80) >> 3);
      v32 = v31 >> 2;
      if ( v31 >> 2 > 0 )
      {
        while ( *(_BYTE *)v27 || v28 != *(_DWORD *)(v27 + 8) )
        {
          if ( !*(_BYTE *)(v27 + 40) && v28 == *(_DWORD *)(v27 + 48) )
          {
            v27 += 40;
            goto LABEL_40;
          }
          if ( !*(_BYTE *)(v27 + 80) && v28 == *(_DWORD *)(v27 + 88) )
          {
            v27 += 80;
            goto LABEL_40;
          }
          if ( !*(_BYTE *)(v27 + 120) && v28 == *(_DWORD *)(v27 + 128) )
          {
            v27 += 120;
            goto LABEL_40;
          }
          v27 += 160;
          if ( !--v32 )
          {
            v31 = 0xCCCCCCCCCCCCCCCDLL * ((v29 - v27) >> 3);
            goto LABEL_30;
          }
        }
        goto LABEL_40;
      }
LABEL_30:
      switch ( v31 )
      {
        case 2LL:
          goto LABEL_35;
        case 3LL:
          if ( !*(_BYTE *)v27 && v28 == *(_DWORD *)(v27 + 8) )
            goto LABEL_40;
          v27 += 40;
LABEL_35:
          if ( !*(_BYTE *)v27 && v28 == *(_DWORD *)(v27 + 8) )
            goto LABEL_40;
          v27 += 40;
          goto LABEL_38;
        case 1LL:
          goto LABEL_38;
      }
LABEL_41:
      if ( v25 == ++v26 )
        goto LABEL_42;
    }
LABEL_38:
    if ( *(_BYTE *)v27 || v28 != *(_DWORD *)(v27 + 8) )
      goto LABEL_41;
LABEL_40:
    if ( v27 == v29 || (unsigned __int8)sub_2ED41E0(a1, v19, v28) )
      goto LABEL_41;
    v34 = *(_DWORD **)(v19 + 32);
    v35 = v34 + 10;
    if ( *(_WORD *)(v19 + 68) != 14 )
    {
      v35 = &v34[10 * (*(_DWORD *)(v19 + 40) & 0xFFFFFF)];
      v34 += 20;
    }
    for ( i = v34; v35 != i; i += 10 )
    {
      if ( !*(_BYTE *)i )
      {
        sub_2EAB0C0((__int64)i, 0);
        *i &= 0xFFF000FF;
      }
    }
LABEL_42:
    v42 += 32;
    result = v42;
  }
  while ( v38 != v42 );
  return result;
}
