// Function: sub_33AEA10
// Address: 0x33aea10
//
__int64 __fastcall sub_33AEA10(unsigned __int8 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r15d
  int v7; // edx
  int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 result; // rax
  __int64 v16; // rax
  unsigned int v17; // edx
  int v18; // ecx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r10
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r11
  unsigned __int64 v25; // rdx
  __int64 *v26; // rax
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-50h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  int v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v7 = *a1;
  v30 = *(_QWORD *)(a4 + 864);
  v8 = v7 - 29;
  if ( v7 != 40 )
    goto LABEL_2;
LABEL_16:
  v9 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_17;
  while ( 1 )
  {
    v10 = sub_BD2BC0((__int64)a1);
    v32 = v11 + v10;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v32 >> 4) )
LABEL_25:
        BUG();
LABEL_17:
      v14 = 0;
      goto LABEL_11;
    }
    if ( !(unsigned int)((v32 - sub_BD2BC0((__int64)a1)) >> 4) )
      goto LABEL_17;
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_25;
    v33 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
    if ( (a1[7] & 0x80u) == 0 )
      BUG();
    v12 = sub_BD2BC0((__int64)a1);
    v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v33);
LABEL_11:
    result = (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v9 - v14) >> 5;
    if ( a2 >= (unsigned int)result )
      return result;
    v16 = sub_338B750(a4, *(_QWORD *)&a1[32 * (a2 - (unsigned __int64)(*((_DWORD *)a1 + 1) & 0x7FFFFFF))]);
    v18 = *(_DWORD *)(v16 + 24);
    if ( v18 == 39 || v18 == 15 )
    {
      v28 = *(_QWORD *)(v16 + 48) + 16LL * v17;
      LOWORD(v4) = *(_WORD *)v28;
      v21 = sub_33EDBD0(v30, *(unsigned int *)(v16 + 96), v4, *(_QWORD *)(v28 + 8), 1);
      v22 = *(unsigned int *)(a3 + 8);
      v24 = v29;
      v25 = v22 + 1;
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
LABEL_19:
        v34 = v21;
        v35 = v24;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v25, 0x10u, v19, v20);
        v22 = *(unsigned int *)(a3 + 8);
        v21 = v34;
        v24 = v35;
      }
    }
    else
    {
      v21 = sub_338B750(a4, *(_QWORD *)&a1[32 * (a2 - (unsigned __int64)(*((_DWORD *)a1 + 1) & 0x7FFFFFF))]);
      v22 = *(unsigned int *)(a3 + 8);
      v24 = v23;
      v25 = v22 + 1;
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        goto LABEL_19;
    }
    v26 = (__int64 *)(*(_QWORD *)a3 + 16 * v22);
    ++a2;
    *v26 = v21;
    v26[1] = v24;
    ++*(_DWORD *)(a3 + 8);
    v27 = *a1;
    v8 = v27 - 29;
    if ( v27 == 40 )
      goto LABEL_16;
LABEL_2:
    v9 = 0;
    if ( v8 != 56 )
    {
      if ( v8 != 5 )
        BUG();
      v9 = 64;
    }
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_17;
  }
}
