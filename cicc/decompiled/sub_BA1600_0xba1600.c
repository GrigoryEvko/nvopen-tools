// Function: sub_BA1600
// Address: 0xba1600
//
__int64 __fastcall sub_BA1600(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v4; // r14d
  unsigned int i; // r12d
  __int64 *v6; // rbx
  __int64 v7; // rsi
  unsigned int v8; // r12d
  __int64 *v9; // rcx
  __int64 v10; // r11
  __int64 v11; // r10
  __int64 result; // rax
  int v13; // eax
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 *v16; // rdi
  __int64 v17; // rcx
  int v18; // r9d
  __int64 *v19; // rsi
  __int64 *v20; // r13
  unsigned int v21; // r9d
  int v22; // esi
  int v23; // eax
  int v24; // eax
  int v25; // [rsp+48h] [rbp-F8h]
  __int64 v26; // [rsp+48h] [rbp-F8h]
  __int64 v27; // [rsp+50h] [rbp-F0h]
  int v28; // [rsp+50h] [rbp-F0h]
  __int64 v29; // [rsp+58h] [rbp-E8h] BYREF
  __int64 *v30; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v32; // [rsp+70h] [rbp-D0h] BYREF
  int v33; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v34; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v35[4]; // [rsp+88h] [rbp-B8h] BYREF
  __int64 v36[3]; // [rsp+A8h] [rbp-98h] BYREF
  __int64 v37[7]; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v38[9]; // [rsp+F8h] [rbp-48h] BYREF

  v2 = a2;
  v29 = a1;
  sub_AF54F0((__int64)&v30, a1);
  v27 = *(_QWORD *)(a2 + 8);
  v25 = *(_DWORD *)(a2 + 24);
  if ( !v25 )
  {
LABEL_22:
    ++*(_QWORD *)v2;
    v21 = 0;
    v30 = 0;
    goto LABEL_23;
  }
  v4 = 1;
  for ( i = (v25 - 1) & sub_AFADE0(&v31, &v32, &v33, v35, &v34, v36, v37, v38); ; i = (v25 - 1) & v8 )
  {
    v6 = (__int64 *)(v27 + 8LL * i);
    v7 = *v6;
    if ( *v6 == -4096 )
    {
LABEL_12:
      v2 = a2;
      v10 = *(_QWORD *)(a2 + 8);
      LODWORD(v11) = *(_DWORD *)(a2 + 24);
      goto LABEL_13;
    }
    if ( v7 != -8192 )
      break;
LABEL_7:
    if ( v7 == -4096 )
      goto LABEL_12;
    v8 = v4 + i;
    ++v4;
  }
  if ( !sub_AF52E0((int *)&v30, v7) )
  {
    v7 = *v6;
    goto LABEL_7;
  }
  v9 = (__int64 *)(v27 + 8LL * i);
  v2 = a2;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(unsigned int *)(a2 + 24);
  if ( v9 != (__int64 *)(v10 + 8 * v11) )
  {
    result = *v9;
    if ( *v9 )
      return result;
  }
LABEL_13:
  v26 = v10;
  v28 = v11;
  if ( !(_DWORD)v11 )
    goto LABEL_22;
  sub_AF54F0((__int64)&v30, v29);
  v13 = sub_AFADE0(&v31, &v32, &v33, v35, &v34, v36, v37, v38);
  v14 = v29;
  LODWORD(v15) = (v28 - 1) & v13;
  v16 = (__int64 *)(v26 + 8LL * (unsigned int)v15);
  result = v29;
  v17 = *v16;
  if ( *v16 == v29 )
    return result;
  v18 = 1;
  v19 = 0;
  while ( v17 != -4096 )
  {
    if ( v17 != -8192 || v19 )
      v16 = v19;
    v15 = (v28 - 1) & (unsigned int)(v15 + v18);
    v20 = (__int64 *)(v26 + 8 * v15);
    v17 = *v20;
    if ( *v20 == v29 )
      return result;
    ++v18;
    v19 = v16;
    v16 = (__int64 *)(v26 + 8 * v15);
  }
  v24 = *(_DWORD *)(v2 + 16);
  v21 = *(_DWORD *)(v2 + 24);
  if ( !v19 )
    v19 = v16;
  ++*(_QWORD *)v2;
  v23 = v24 + 1;
  v30 = v19;
  if ( 4 * v23 >= 3 * v21 )
  {
LABEL_23:
    v22 = 2 * v21;
    goto LABEL_24;
  }
  if ( v21 - (v23 + *(_DWORD *)(v2 + 20)) > v21 >> 3 )
    goto LABEL_25;
  v22 = v21;
LABEL_24:
  sub_B06170(v2, v22);
  sub_AFD7F0(v2, &v29, &v30);
  v19 = v30;
  v14 = v29;
  v23 = *(_DWORD *)(v2 + 16) + 1;
LABEL_25:
  *(_DWORD *)(v2 + 16) = v23;
  if ( *v19 != -4096 )
    --*(_DWORD *)(v2 + 20);
  *v19 = v14;
  return v29;
}
