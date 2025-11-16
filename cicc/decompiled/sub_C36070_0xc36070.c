// Function: sub_C36070
// Address: 0xc36070
//
unsigned int *__fastcall sub_C36070(__int64 a1, char a2, char a3, unsigned __int64 *a4)
{
  __int64 v6; // r12
  unsigned int v7; // r14d
  unsigned int *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rsi
  __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned int *result; // rax
  int v17; // r15d
  __int64 v18; // rsi
  unsigned int v19; // eax
  unsigned __int64 v20; // rax
  unsigned int v21; // esi
  unsigned __int64 v22; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-38h]

  if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) == 2 || a3 && !*(_BYTE *)(*(_QWORD *)a1 + 25LL) )
    BUG();
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF0 | 1 | (8 * (a3 & 1));
  *(_DWORD *)(a1 + 16) = sub_C36030((unsigned int **)a1);
  v6 = sub_C33900(a1);
  v23 = 1;
  v7 = sub_C337D0(a1);
  v8 = *(unsigned int **)a1;
  v22 = 0;
  if ( v8[4] == 1 )
  {
    if ( v8[5] == 2 )
    {
      *(_BYTE *)(a1 + 20) |= 8u;
      v21 = v8[2];
      v20 = 0;
      LODWORD(v9) = v21 - 1;
      v25 = v9;
      if ( (unsigned int)v9 <= 0x40 )
        goto LABEL_30;
      sub_C43690(&v24, 0, 0);
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
    }
    else
    {
      v19 = v8[2];
      LODWORD(v9) = v19 - 1;
      v25 = v19 - 1;
      if ( v19 - 1 <= 0x40 )
      {
        v20 = 0xFFFFFFFFFFFFFFFFLL >> ~((unsigned __int8)v19 - 2);
        if ( !(_DWORD)v9 )
          v20 = 0;
        goto LABEL_30;
      }
      sub_C43690(&v24, -1, 1);
    }
    v20 = v24;
    LODWORD(v9) = v25;
LABEL_30:
    v22 = v20;
    a4 = &v22;
    a2 = 0;
    v23 = v9;
    goto LABEL_7;
  }
  if ( !a4 )
  {
    sub_C45D00(v6, 0, v7);
    goto LABEL_15;
  }
  LODWORD(v9) = *((_DWORD *)a4 + 2);
LABEL_7:
  v10 = ((unsigned __int64)(unsigned int)v9 + 63) >> 6;
  v11 = (unsigned int)v10;
  if ( v7 > (unsigned int)v10 )
  {
    sub_C45D00(v6, 0, v7);
    v9 = *((unsigned int *)a4 + 2);
    v11 = (unsigned __int64)(v9 + 63) >> 6;
  }
  if ( v7 <= (unsigned int)v11 )
    v11 = v7;
  if ( (unsigned int)v9 <= 0x40 )
    v12 = a4;
  else
    v12 = (_QWORD *)*a4;
  sub_C45D30(v6, v12, v11);
  v13 = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) - 1) >> 6;
  v14 = v13 + 1;
  *(_QWORD *)(v6 + 8 * v13) &= ~(-1LL << (*(_BYTE *)(*(_QWORD *)a1 + 8LL) - 1));
  if ( v7 != (_DWORD)v13 + 1 )
  {
    do
    {
      v15 = v14++;
      *(_QWORD *)(v6 + 8 * v15) = 0;
    }
    while ( v7 != v14 );
  }
LABEL_15:
  result = *(unsigned int **)a1;
  v17 = 2;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) >= 2u )
    v17 = result[2];
  v18 = (unsigned int)(v17 - 2);
  if ( a2 )
  {
    sub_C45DD0(v6, v18);
    if ( (unsigned __int8)sub_C45D60(v6, v7) )
    {
      sub_C45DB0(v6, (unsigned int)(v17 - 3));
      result = *(unsigned int **)a1;
      if ( *(_UNKNOWN **)a1 != &unk_3F655E0 )
        goto LABEL_22;
      goto LABEL_33;
    }
  }
  else
  {
    if ( result[5] == 2 )
      goto LABEL_21;
    sub_C45DB0(v6, v18);
  }
  result = *(unsigned int **)a1;
LABEL_21:
  if ( result != (unsigned int *)&unk_3F655E0 )
    goto LABEL_22;
LABEL_33:
  result = (unsigned int *)sub_C45DB0(v6, (unsigned int)(v17 - 1));
LABEL_22:
  if ( v23 > 0x40 )
  {
    if ( v22 )
      return (unsigned int *)j_j___libc_free_0_0(v22);
  }
  return result;
}
