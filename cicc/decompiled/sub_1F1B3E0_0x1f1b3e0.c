// Function: sub_1F1B3E0
// Address: 0x1f1b3e0
//
__int64 __fastcall sub_1F1B3E0(__int64 a1, int a2, int *a3)
{
  __int64 v4; // rdi
  unsigned int v6; // esi
  int v7; // r13d
  int v8; // r10d
  int *v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned int i; // ecx
  int *v15; // r14
  int v16; // r11d
  unsigned int v17; // ecx
  __int64 v18; // rax
  unsigned __int64 v19; // r15
  __int64 v20; // r13
  unsigned __int64 v21; // rdx
  int v22; // r9d
  unsigned int v23; // eax
  __int64 v24; // r8
  __int64 v25; // r10
  __int64 v26; // rbx
  __int64 result; // rax
  unsigned int v28; // ebx
  __int64 v29; // rcx
  int v30; // edx
  int v31; // ecx
  __int64 v32; // rax
  _QWORD *v33; // rsi
  _QWORD *v34; // rax
  __int64 v35; // rdx
  int v36; // ecx
  int v37; // ecx
  int *v38; // rdi
  __int64 v39; // rdx
  int v40; // r8d
  unsigned __int64 v41; // rsi
  unsigned __int64 v42; // rsi
  unsigned int j; // eax
  int v44; // esi
  unsigned int v45; // eax
  int v46; // edx
  int v47; // edx
  int v48; // r8d
  __int64 v49; // rsi
  unsigned int k; // eax
  int v51; // ecx
  unsigned int v52; // eax
  __int64 v53; // [rsp+0h] [rbp-50h]
  int v54; // [rsp+14h] [rbp-3Ch]
  __int64 v55; // [rsp+18h] [rbp-38h]
  int v56; // [rsp+18h] [rbp-38h]

  v4 = a1 + 400;
  v6 = *(_DWORD *)(a1 + 424);
  v7 = *a3;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 400);
    goto LABEL_39;
  }
  v8 = 1;
  v9 = 0;
  v10 = *(_QWORD *)(a1 + 408);
  v11 = ((((unsigned int)(37 * v7) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v7) << 32)) >> 22)
      ^ (((unsigned int)(37 * v7) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v7) << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  v13 = ((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - (v12 << 27));
  for ( i = v13 & (v6 - 1); ; i = (v6 - 1) & v17 )
  {
    v15 = (int *)(v10 + 16LL * i);
    v16 = *v15;
    if ( a2 == *v15 && v7 == v15[1] )
    {
      v18 = *((_QWORD *)v15 + 1);
      v19 = v18 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_26;
      v20 = *(_QWORD *)(a1 + 16);
      v21 = *(unsigned int *)(v20 + 408);
      v22 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                      + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + a2));
      v23 = v22 & 0x7FFFFFFF;
      v24 = v22 & 0x7FFFFFFF;
      v25 = 8 * v24;
      if ( (v22 & 0x7FFFFFFFu) < (unsigned int)v21 )
      {
        v26 = *(_QWORD *)(*(_QWORD *)(v20 + 400) + 8LL * v23);
        if ( v26 )
          goto LABEL_14;
      }
      v28 = v23 + 1;
      if ( (unsigned int)v21 < v23 + 1 )
      {
        v32 = v28;
        if ( v28 >= v21 )
        {
          if ( v28 > v21 )
          {
            if ( v28 > (unsigned __int64)*(unsigned int *)(v20 + 412) )
            {
              v53 = v22 & 0x7FFFFFFF;
              v54 = v22;
              sub_16CD150(v20 + 400, (const void *)(v20 + 416), v28, 8, v24, v22);
              v21 = *(unsigned int *)(v20 + 408);
              v24 = v53;
              v25 = 8 * v53;
              v22 = v54;
              v32 = v28;
            }
            v29 = *(_QWORD *)(v20 + 400);
            v33 = (_QWORD *)(v29 + 8 * v32);
            v34 = (_QWORD *)(v29 + 8 * v21);
            v35 = *(_QWORD *)(v20 + 416);
            if ( v33 != v34 )
            {
              do
                *v34++ = v35;
              while ( v33 != v34 );
              v29 = *(_QWORD *)(v20 + 400);
            }
            *(_DWORD *)(v20 + 408) = v28;
            goto LABEL_17;
          }
        }
        else
        {
          *(_DWORD *)(v20 + 408) = v28;
        }
      }
      v29 = *(_QWORD *)(v20 + 400);
LABEL_17:
      v55 = v24;
      *(_QWORD *)(v29 + v25) = sub_1DBA290(v22);
      v26 = *(_QWORD *)(*(_QWORD *)(v20 + 400) + 8 * v55);
      sub_1DBB110((_QWORD *)v20, v26);
LABEL_14:
      result = sub_1F15470((_QWORD *)a1, v26, v19, 0);
      *((_QWORD *)v15 + 1) = 4;
      return result;
    }
    if ( v16 == -1 )
      break;
    if ( v16 == -2 && v15[1] == -2 && !v9 )
      v9 = (int *)(v10 + 16LL * i);
LABEL_9:
    v17 = v8 + i;
    ++v8;
  }
  if ( v15[1] != -1 )
    goto LABEL_9;
  v30 = *(_DWORD *)(a1 + 416);
  if ( v9 )
    v15 = v9;
  ++*(_QWORD *)(a1 + 400);
  v31 = v30 + 1;
  if ( 4 * (v30 + 1) >= 3 * v6 )
  {
LABEL_39:
    sub_1F1A4C0(v4, 2 * v6);
    v36 = *(_DWORD *)(a1 + 424);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = 0;
      v40 = 1;
      v41 = ((((unsigned int)(37 * v7) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v7) << 32)) >> 22)
          ^ (((unsigned int)(37 * v7) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v7) << 32));
      v42 = ((9 * (((v41 - 1 - (v41 << 13)) >> 8) ^ (v41 - 1 - (v41 << 13)))) >> 15)
          ^ (9 * (((v41 - 1 - (v41 << 13)) >> 8) ^ (v41 - 1 - (v41 << 13))));
      for ( j = v37 & (((v42 - 1 - (v42 << 27)) >> 31) ^ (v42 - 1 - ((_DWORD)v42 << 27))); ; j = v37 & v45 )
      {
        v39 = *(_QWORD *)(a1 + 408);
        v15 = (int *)(v39 + 16LL * j);
        v44 = *v15;
        if ( a2 == *v15 && v7 == v15[1] )
          break;
        if ( v44 == -1 )
        {
          if ( v15[1] == -1 )
          {
LABEL_62:
            if ( v38 )
              v15 = v38;
            v31 = *(_DWORD *)(a1 + 416) + 1;
            goto LABEL_23;
          }
        }
        else if ( v44 == -2 && v15[1] == -2 && !v38 )
        {
          v38 = (int *)(v39 + 16LL * j);
        }
        v45 = v40 + j;
        ++v40;
      }
      goto LABEL_58;
    }
LABEL_67:
    ++*(_DWORD *)(a1 + 416);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 420) - v31 <= v6 >> 3 )
  {
    v56 = v13;
    sub_1F1A4C0(v4, v6);
    v46 = *(_DWORD *)(a1 + 424);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = 1;
      v38 = 0;
      for ( k = v47 & v56; ; k = v47 & v52 )
      {
        v49 = *(_QWORD *)(a1 + 408);
        v15 = (int *)(v49 + 16LL * k);
        v51 = *v15;
        if ( a2 == *v15 && v7 == v15[1] )
          break;
        if ( v51 == -1 )
        {
          if ( v15[1] == -1 )
            goto LABEL_62;
        }
        else if ( v51 == -2 && v15[1] == -2 && !v38 )
        {
          v38 = (int *)(v49 + 16LL * k);
        }
        v52 = v48 + k;
        ++v48;
      }
LABEL_58:
      v31 = *(_DWORD *)(a1 + 416) + 1;
      goto LABEL_23;
    }
    goto LABEL_67;
  }
LABEL_23:
  *(_DWORD *)(a1 + 416) = v31;
  if ( *v15 != -1 || v15[1] != -1 )
    --*(_DWORD *)(a1 + 420);
  *v15 = a2;
  v18 = 0;
  v15[1] = v7;
  *((_QWORD *)v15 + 1) = 0;
LABEL_26:
  result = v18 | 4;
  *((_QWORD *)v15 + 1) = result;
  return result;
}
