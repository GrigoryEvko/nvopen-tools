// Function: sub_2990B70
// Address: 0x2990b70
//
__int64 __fastcall sub_2990B70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // r11
  __int64 v7; // rdi
  int v8; // r9d
  unsigned int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // r12
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  _QWORD *v28; // rdx
  __int64 *v29; // r8
  __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rsi
  int v33; // eax
  unsigned __int64 *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  char *v37; // r15
  int v38; // ebx
  int v39; // edx
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  char v43; // [rsp+24h] [rbp-5Ch]
  __int64 v44; // [rsp+28h] [rbp-58h] BYREF
  __int64 v45[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v46; // [rsp+40h] [rbp-40h]

  v4 = a1 + 544;
  v5 = *(_DWORD *)(a1 + 568);
  v44 = a3;
  if ( !v5 )
  {
    v45[0] = 0;
    ++*(_QWORD *)(a1 + 544);
LABEL_53:
    v5 *= 2;
    goto LABEL_54;
  }
  v6 = *(_QWORD *)(a1 + 552);
  v7 = a3;
  v8 = 1;
  v9 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v10 = 0;
  v11 = v6 + 56LL * v9;
  v12 = *(_QWORD *)v11;
  if ( v7 == *(_QWORD *)v11 )
  {
LABEL_3:
    v13 = v11 + 8;
    goto LABEL_4;
  }
  while ( v12 != -4096 )
  {
    if ( !v10 && v12 == -8192 )
      v10 = v11;
    v9 = (v5 - 1) & (v8 + v9);
    v11 = v6 + 56LL * v9;
    v12 = *(_QWORD *)v11;
    if ( v7 == *(_QWORD *)v11 )
      goto LABEL_3;
    ++v8;
  }
  if ( v10 )
    v11 = v10;
  ++*(_QWORD *)(a1 + 544);
  v38 = *(_DWORD *)(a1 + 560);
  v45[0] = v11;
  v39 = v38 + 1;
  if ( 4 * (v38 + 1) >= 3 * v5 )
    goto LABEL_53;
  if ( v5 - *(_DWORD *)(a1 + 564) - v39 <= v5 >> 3 )
  {
LABEL_54:
    sub_298C3B0(v4, v5);
    sub_298B940(v4, &v44, v45);
    v7 = v44;
    v39 = *(_DWORD *)(a1 + 560) + 1;
    v11 = v45[0];
  }
  *(_DWORD *)(a1 + 560) = v39;
  if ( *(_QWORD *)v11 != -4096 )
    --*(_DWORD *)(a1 + 564);
  *(_QWORD *)v11 = v7;
  v13 = v11 + 8;
  *(_QWORD *)(v11 + 40) = v11 + 56;
  v7 = v44;
  *(_QWORD *)(v11 + 48) = 0;
  *(_OWORD *)(v11 + 8) = 0;
  *(_OWORD *)(v11 + 24) = 0;
LABEL_4:
  result = sub_AA5930(v7);
  v40 = v15;
  v16 = result;
  while ( v40 != v16 )
  {
    v43 = 0;
    while ( 1 )
    {
      v17 = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
      if ( !v17 )
        break;
      while ( 1 )
      {
        v18 = 0;
        v19 = *(_QWORD *)(v16 - 8) + 32LL * *(unsigned int *)(v16 + 72);
        while ( a2 != *(_QWORD *)(v19 + 8 * v18) )
        {
          if ( v17 == (_DWORD)++v18 )
            goto LABEL_29;
        }
        v20 = 0;
        while ( 1 )
        {
          v21 = v20;
          if ( a2 == *(_QWORD *)(v19 + 8 * v20) )
            break;
          if ( v17 == (_DWORD)++v20 )
          {
            v21 = -1;
            break;
          }
        }
        v22 = sub_B48BF0(v16, v21, 0);
        v45[0] = v16;
        v23 = v22;
        v24 = sub_29908D0(v13, v45);
        v27 = *(unsigned int *)(v24 + 8);
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v24 + 12) )
        {
          v41 = v24;
          sub_C8D5F0(v24, (const void *)(v24 + 16), v27 + 1, 0x10u, v25, v26);
          v24 = v41;
          v27 = *(unsigned int *)(v41 + 8);
        }
        v28 = (_QWORD *)(*(_QWORD *)v24 + 16 * v27);
        *v28 = a2;
        v28[1] = v23;
        ++*(_DWORD *)(v24 + 8);
        if ( v43 )
          break;
        v45[0] = 4;
        v45[1] = 0;
        v46 = v16;
        if ( v16 != -8192 && v16 != -4096 )
          sub_BD73F0((__int64)v45);
        v29 = v45;
        v30 = *(unsigned int *)(a1 + 344);
        v31 = *(_QWORD *)(a1 + 336);
        v32 = v30 + 1;
        v33 = *(_DWORD *)(a1 + 344);
        if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 348) )
        {
          v36 = a1 + 336;
          if ( v31 > (unsigned __int64)v45 || (unsigned __int64)v45 >= v31 + 24 * v30 )
          {
            sub_D6B130(v36, v32, v30, v31, (__int64)v45, v26);
            v29 = v45;
            v30 = *(unsigned int *)(a1 + 344);
            v31 = *(_QWORD *)(a1 + 336);
            v33 = *(_DWORD *)(a1 + 344);
          }
          else
          {
            v37 = (char *)v45 - v31;
            sub_D6B130(v36, v32, v30, v31, (__int64)v45, v26);
            v31 = *(_QWORD *)(a1 + 336);
            v30 = *(unsigned int *)(a1 + 344);
            v29 = (__int64 *)&v37[v31];
            v33 = *(_DWORD *)(a1 + 344);
          }
        }
        v34 = (unsigned __int64 *)(v31 + 24 * v30);
        if ( v34 )
        {
          *v34 = 4;
          v35 = v29[2];
          v34[1] = 0;
          v34[2] = v35;
          if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
            sub_BD6050(v34, *v29 & 0xFFFFFFFFFFFFFFF8LL);
          v33 = *(_DWORD *)(a1 + 344);
        }
        *(_DWORD *)(a1 + 344) = v33 + 1;
        if ( v46 != -4096 && v46 != 0 && v46 != -8192 )
          sub_BD60C0(v45);
        v43 = 1;
        v17 = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
        if ( !v17 )
          goto LABEL_29;
      }
    }
LABEL_29:
    result = *(_QWORD *)(v16 + 32);
    if ( !result )
      BUG();
    v16 = 0;
    if ( *(_BYTE *)(result - 24) == 84 )
      v16 = result - 24;
  }
  return result;
}
