// Function: sub_F59AD0
// Address: 0xf59ad0
//
__int64 __fastcall sub_F59AD0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r15d
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v9; // rdi
  __int64 *v10; // rdi
  __int64 *v11; // rsi
  int v12; // eax
  __int64 v13; // r14
  int v14; // ecx
  int v15; // r9d
  unsigned int v16; // ebx
  __int64 *v17; // r8
  __int64 *v18; // rdx
  __int64 v19; // rsi
  char v20; // al
  unsigned int v21; // ebx
  __int64 *v22; // [rsp+10h] [rbp-50h]
  int v23; // [rsp+18h] [rbp-48h]
  int v24; // [rsp+1Ch] [rbp-44h]
  unsigned __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = 32LL * *(unsigned int *)(*a2 + 72);
  v26[0] = sub_F59720(
             (_QWORD *)(*(_QWORD *)(v6 - 8) + v9),
             *(_QWORD *)(v6 - 8) + v9 + 8LL * (*(_DWORD *)(*a2 + 4) & 0x7FFFFFF));
  if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
  {
    v10 = *(__int64 **)(v6 - 8);
    v11 = &v10[4 * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)];
  }
  else
  {
    v11 = (__int64 *)v6;
    v10 = (__int64 *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  }
  v25 = sub_F58E90(v10, v11);
  v12 = sub_C41E80((__int64 *)&v25, (__int64 *)v26);
  v13 = *a2;
  v14 = v4 - 1;
  v15 = 1;
  v16 = (v4 - 1) & v12;
  v17 = 0;
  while ( 1 )
  {
    v18 = (__int64 *)(v7 + 8LL * v16);
    v19 = *v18;
    if ( *v18 == -8192 || *v18 == -4096 || v13 == -4096 || v13 == -8192 )
      break;
    v23 = v15;
    v22 = v17;
    v24 = v14;
    v20 = sub_B46220(v13, v19);
    v14 = v24;
    v17 = v22;
    v15 = v23;
    v18 = (__int64 *)(v7 + 8LL * v16);
    if ( v20 )
    {
LABEL_9:
      *a3 = v18;
      return 1;
    }
LABEL_12:
    v21 = v15 + v16;
    ++v15;
    v16 = v14 & v21;
  }
  if ( v19 == v13 )
    goto LABEL_9;
  if ( v19 != -4096 )
  {
    if ( !v17 && *v18 == -8192 )
      v17 = (__int64 *)(v7 + 8LL * v16);
    goto LABEL_12;
  }
  if ( !v17 )
    v17 = (__int64 *)(v7 + 8LL * v16);
  *a3 = v17;
  return 0;
}
