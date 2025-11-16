// Function: sub_25DCB60
// Address: 0x25dcb60
//
__int64 __fastcall sub_25DCB60(__int64 a1, __int64 (__fastcall *a2)(__int64, __int64), __int64 a3, unsigned int a4)
{
  __int64 v5; // rsi
  __int64 *v9; // rdx
  __int64 v10; // rdx
  char *v11; // rsi
  unsigned __int64 v12; // rdx
  _BYTE *v13; // rax
  __int64 v14; // r12
  __int64 *v15; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // ecx
  int *v20; // rdx
  int v21; // esi
  int v22; // edx
  int v23; // r9d
  unsigned int v24[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)(a1 + 32);
  if ( v5 == a1 + 24 )
    return 0;
  if ( v5 )
    v5 -= 56;
  v9 = (__int64 *)a2(a3, v5);
  if ( (v9[2] & (1LL << a4)) != 0 )
    return 0;
  v10 = *v9;
  if ( (((int)*(unsigned __int8 *)(v10 + 22) >> (2 * (a4 & 3))) & 3) == 0 )
    return 0;
  if ( (((int)*(unsigned __int8 *)(v10 + 22) >> (2 * (a4 & 3))) & 3) != 3 )
  {
    v17 = *(unsigned int *)(v10 + 160);
    v18 = *(_QWORD *)(v10 + 144);
    if ( (_DWORD)v17 )
    {
      v19 = (v17 - 1) & (37 * a4);
      v20 = (int *)(v18 + 40LL * v19);
      v21 = *v20;
      if ( a4 == *v20 )
      {
LABEL_15:
        v11 = (char *)*((_QWORD *)v20 + 1);
        v12 = *((_QWORD *)v20 + 2);
        goto LABEL_8;
      }
      v22 = 1;
      while ( v21 != -1 )
      {
        v23 = v22 + 1;
        v19 = (v17 - 1) & (v22 + v19);
        v20 = (int *)(v18 + 40LL * v19);
        v21 = *v20;
        if ( a4 == *v20 )
          goto LABEL_15;
        v22 = v23;
      }
    }
    v20 = (int *)(v18 + 40 * v17);
    goto LABEL_15;
  }
  v11 = (&off_4977320)[2 * a4];
  v12 = qword_4977328[2 * a4];
LABEL_8:
  v13 = sub_BA8CB0(a1, (__int64)v11, v12);
  v14 = (__int64)v13;
  if ( !v13 )
    return 0;
  v15 = (__int64 *)a2(a3, (__int64)v13);
  if ( !sub_981210(*v15, v14, v24) || v24[0] != a4 )
    return 0;
  return v14;
}
