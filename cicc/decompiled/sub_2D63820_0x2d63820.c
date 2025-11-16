// Function: sub_2D63820
// Address: 0x2d63820
//
__int64 __fastcall sub_2D63820(__int64 a1, __int64 a2, __int64 a3, int *a4, __int64 a5, __int64 a6, __int64 *a7)
{
  _BYTE **v9; // rdx
  _BYTE *v10; // r14
  __int64 *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r12
  unsigned __int64 *v18; // rdx
  __int64 v19; // rsi
  char v20; // [rsp+7h] [rbp-39h]

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v9 = *(_BYTE ***)(a1 - 8);
    v10 = *v9;
    if ( **v9 != 68 )
      goto LABEL_3;
  }
  else
  {
    v9 = (_BYTE **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v10 = *v9;
    if ( **v9 != 68 )
    {
LABEL_3:
      if ( (v10[7] & 0x40) != 0 )
        v11 = (__int64 *)*((_QWORD *)v10 - 1);
      else
        v11 = (__int64 *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v12 = a1;
      sub_2D598D0(a2, a1, 0, *v11);
      *a4 = 0;
      v20 = 0;
      if ( *((_QWORD *)v10 + 2) )
        goto LABEL_6;
LABEL_18:
      sub_2D5CED0(a2, (__int64)v10, 0);
      goto LABEL_6;
    }
  }
  v20 = sub_2D5C100(a7, v10, (__int64)v9, (__int64)a4, a5) ^ 1;
  if ( (v10[7] & 0x40) != 0 )
    v18 = (unsigned __int64 *)*((_QWORD *)v10 - 1);
  else
    v18 = (unsigned __int64 *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
  v12 = sub_2D5EBE0(a2, a1, *v18, *(__int64 ***)(a1 + 8));
  sub_2D58400(a2, a1, v12);
  sub_2D5CED0(a2, a1, 0);
  *a4 = 0;
  if ( !*((_QWORD *)v10 + 2) )
    goto LABEL_18;
LABEL_6:
  if ( *(_BYTE *)v12 <= 0x1Cu )
    return v12;
  v14 = *(_QWORD *)(v12 + 8);
  if ( (*(_BYTE *)(v12 + 7) & 0x40) == 0 )
  {
    v15 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
    v16 = *(_QWORD *)v15;
    if ( v14 != *(_QWORD *)(*(_QWORD *)v15 + 8LL) )
      goto LABEL_9;
LABEL_20:
    v19 = v12;
    v12 = v16;
    sub_2D5CED0(a2, v19, v16);
    return v12;
  }
  v15 = *(_QWORD *)(v12 - 8);
  v16 = *(_QWORD *)v15;
  if ( v14 == *(_QWORD *)(*(_QWORD *)v15 + 8LL) )
    goto LABEL_20;
LABEL_9:
  if ( a5 )
    sub_9C95B0(a5, v12);
  *a4 = (unsigned __int8)(sub_2D5C100(a7, (unsigned __int8 *)v12, v14, v15, v13) | v20) ^ 1;
  return v12;
}
