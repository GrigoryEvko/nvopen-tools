// Function: sub_1C0A960
// Address: 0x1c0a960
//
__int64 __fastcall sub_1C0A960(int *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v6; // rcx
  int v7; // r8d
  int v9; // r10d
  unsigned int v10; // edx
  unsigned int v11; // r11d
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r9
  __int64 v15; // r15
  int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r12
  int v21; // eax
  _QWORD *v22; // r8
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // ecx
  unsigned int v27; // esi
  int v28; // edx
  int v29; // edx
  int v30; // r12d
  __int64 *v31; // r10
  int v32; // r11d
  __int64 *v33; // r9
  int v34; // eax
  int v35; // edx
  __int64 v36; // rcx
  __int64 *v37; // r10
  __int64 v38; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v39[7]; // [rsp+18h] [rbp-38h] BYREF

  v38 = a2;
  v4 = (unsigned int)a1[16];
  if ( !(_DWORD)v4 )
    goto LABEL_45;
  v6 = *((_QWORD *)a1 + 6);
  v7 = v4 - 1;
  v9 = 1;
  v10 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v10;
  v12 = (__int64 *)(v6 + 16LL * v10);
  v13 = *v12;
  v14 = *v12;
  if ( a2 == *v12 )
  {
    if ( v12 != (__int64 *)(v6 + 16 * v4) )
    {
      v15 = v12[1];
      goto LABEL_5;
    }
LABEL_45:
    BUG();
  }
  while ( 1 )
  {
    if ( v14 == -8 )
      goto LABEL_45;
    v30 = v9 + 1;
    v11 = v7 & (v11 + v9);
    v31 = (__int64 *)(v6 + 16LL * v11);
    v14 = *v31;
    if ( a2 == *v31 )
      break;
    v9 = v30;
  }
  if ( v31 == (__int64 *)(v6 + 16LL * (unsigned int)v4) )
    goto LABEL_45;
  v32 = 1;
  v33 = 0;
  while ( v13 != -8 )
  {
    if ( v13 != -16 || v33 )
      v12 = v33;
    v10 = v7 & (v32 + v10);
    v37 = (__int64 *)(v6 + 16LL * v10);
    v13 = *v37;
    if ( a2 == *v37 )
    {
      v15 = v37[1];
      goto LABEL_5;
    }
    ++v32;
    v33 = v12;
    v12 = (__int64 *)(v6 + 16LL * v10);
  }
  if ( !v33 )
    v33 = v12;
  v34 = a1[14];
  ++*((_QWORD *)a1 + 5);
  v35 = v34 + 1;
  if ( 4 * (v34 + 1) >= (unsigned int)(3 * v4) )
  {
    LODWORD(v4) = 2 * v4;
    goto LABEL_35;
  }
  v36 = a2;
  if ( (int)v4 - a1[15] - v35 <= (unsigned int)v4 >> 3 )
  {
LABEL_35:
    sub_1C04E30((__int64)(a1 + 10), v4);
    sub_1C09800((__int64)(a1 + 10), &v38, v39);
    v33 = (__int64 *)v39[0];
    v36 = v38;
    v35 = a1[14] + 1;
  }
  a1[14] = v35;
  if ( *v33 != -8 )
    --a1[15];
  *v33 = v36;
  v15 = 0;
  v33[1] = 0;
LABEL_5:
  v16 = *(_DWORD *)(v15 + 56);
  if ( !v16 )
  {
LABEL_10:
    v19 = sub_22077B0(24);
    v21 = *a1;
    if ( v19 )
    {
      *(_QWORD *)v19 = a2;
      *(_QWORD *)(v19 + 8) = a3;
      *(_DWORD *)(v19 + 16) = v21;
    }
    LODWORD(v38) = v21;
    v23 = sub_1C09960((__int64)(a1 + 18), (int *)&v38, v39);
    v24 = v39[0];
    if ( (_BYTE)v23 )
      goto LABEL_13;
    v26 = a1[22];
    v27 = a1[24];
    ++*((_QWORD *)a1 + 9);
    v22 = v39;
    v28 = v26 + 1;
    v23 = 2 * v27;
    if ( 4 * (v26 + 1) >= 3 * v27 )
    {
      v27 *= 2;
    }
    else if ( v27 - a1[23] - v28 > v27 >> 3 )
    {
LABEL_18:
      a1[22] = v28;
      if ( *(_DWORD *)v24 != 0x7FFFFFFF )
        --a1[23];
      v29 = v38;
      *(_QWORD *)(v24 + 8) = 0;
      *(_DWORD *)v24 = v29;
LABEL_13:
      *(_QWORD *)(v24 + 8) = v19;
      ++*a1;
      v25 = *(unsigned int *)(v15 + 56);
      if ( (unsigned int)v25 >= *(_DWORD *)(v15 + 60) )
      {
        sub_16CD150(v15 + 48, (const void *)(v15 + 64), 0, 8, (int)v22, v23);
        v25 = *(unsigned int *)(v15 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v15 + 48) + 8 * v25) = v19;
      ++*(_DWORD *)(v15 + 56);
      return v19;
    }
    sub_1C0A790((__int64)(a1 + 18), v27);
    sub_1C09960((__int64)(a1 + 18), (int *)&v38, v39);
    v24 = v39[0];
    v28 = a1[22] + 1;
    goto LABEL_18;
  }
  v17 = *(__int64 **)(v15 + 48);
  v18 = (__int64)&v17[(unsigned int)(v16 - 1) + 1];
  while ( 1 )
  {
    v19 = *v17;
    if ( a3 == *(_QWORD *)(*v17 + 8) )
      return v19;
    if ( ++v17 == (__int64 *)v18 )
      goto LABEL_10;
  }
}
