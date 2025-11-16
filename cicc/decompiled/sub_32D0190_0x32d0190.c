// Function: sub_32D0190
// Address: 0x32d0190
//
__int64 __fastcall sub_32D0190(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  _QWORD *v5; // rdi
  _QWORD *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rbx
  int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // r11
  _QWORD *v20; // rbx
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  unsigned int v24; // esi
  int v25; // ecx
  int v26; // ecx
  unsigned int v27; // edx
  _QWORD *v28; // r10
  __int64 v29; // rdi
  int v30; // eax
  int v31; // eax
  __int64 v32; // rax
  int v33; // ecx
  int v34; // ecx
  unsigned int v35; // edx
  __int64 v36; // rdi
  __int64 v37; // [rsp+8h] [rbp-98h]
  int v38; // [rsp+8h] [rbp-98h]
  int v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  int v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+18h] [rbp-88h]
  const void *v44; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  __int64 v46; // [rsp+38h] [rbp-68h] BYREF
  __int64 v47[4]; // [rsp+40h] [rbp-60h] BYREF
  char v48; // [rsp+60h] [rbp-40h]

  sub_34161C0(*(_QWORD *)a1, a2[2], a2[3], a2[4], a2[5]);
  v45 = a2[4];
  v4 = *(_QWORD *)(v45 + 56);
  v44 = (const void *)(a1 + 56);
  v43 = a1 + 40;
  if ( v4 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v4 + 16);
        if ( *(_DWORD *)(v9 + 24) == 328 )
          goto LABEL_5;
        v10 = *(_DWORD *)(a1 + 584);
        v46 = *(_QWORD *)(v4 + 16);
        if ( !v10 )
          break;
        sub_32B33F0((__int64)v47, a1 + 568, &v46);
        if ( !v48 )
          goto LABEL_4;
        v11 = *(unsigned int *)(a1 + 608);
        v12 = v46;
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
        {
          sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v11 + 1, 8u, v7, v8);
          v11 = *(unsigned int *)(a1 + 608);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8 * v11) = v12;
        ++*(_DWORD *)(a1 + 608);
        if ( *(int *)(v9 + 88) < 0 )
          goto LABEL_12;
LABEL_5:
        v4 = *(_QWORD *)(v4 + 32);
        if ( !v4 )
          goto LABEL_15;
      }
      v5 = *(_QWORD **)(a1 + 600);
      v6 = &v5[*(unsigned int *)(a1 + 608)];
      if ( v6 == sub_325EB50(v5, (__int64)v6, &v46) )
      {
        if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
        {
          sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v7 + 1, 8u, v7, v8);
          v6 = (_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 608));
        }
        *v6 = v9;
        v17 = (unsigned int)(*(_DWORD *)(a1 + 608) + 1);
        *(_DWORD *)(a1 + 608) = v17;
        if ( (unsigned int)v17 > 0x20 )
          break;
      }
LABEL_4:
      if ( *(int *)(v9 + 88) >= 0 )
        goto LABEL_5;
LABEL_12:
      *(_DWORD *)(v9 + 88) = *(_DWORD *)(a1 + 48);
      v13 = *(unsigned int *)(a1 + 48);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(v43, v44, v13 + 1, 8u, v7, v8);
        v13 = *(unsigned int *)(a1 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v13) = v9;
      ++*(_DWORD *)(a1 + 48);
      v4 = *(_QWORD *)(v4 + 32);
      if ( !v4 )
        goto LABEL_15;
    }
    v18 = *(_QWORD **)(a1 + 600);
    v42 = a1 + 568;
    v19 = v9;
    v20 = &v18[v17];
    while ( 1 )
    {
      v24 = *(_DWORD *)(a1 + 592);
      if ( !v24 )
        break;
      v8 = v24 - 1;
      v7 = *(_QWORD *)(a1 + 576);
      v21 = v8 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
      v22 = (_QWORD *)(v7 + 8LL * v21);
      v23 = *v22;
      if ( *v18 != *v22 )
      {
        v39 = 1;
        v28 = 0;
        while ( v23 != -4096 )
        {
          if ( v23 != -8192 || v28 )
            v22 = v28;
          v21 = v8 & (v39 + v21);
          v23 = *(_QWORD *)(v7 + 8LL * v21);
          if ( *v18 == v23 )
            goto LABEL_23;
          ++v39;
          v28 = v22;
          v22 = (_QWORD *)(v7 + 8LL * v21);
        }
        v31 = *(_DWORD *)(a1 + 584);
        if ( !v28 )
          v28 = v22;
        ++*(_QWORD *)(a1 + 568);
        v30 = v31 + 1;
        if ( 4 * v30 < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(a1 + 588) - v30 <= v24 >> 3 )
          {
            v40 = v19;
            sub_32B3220(v42, v24);
            v33 = *(_DWORD *)(a1 + 592);
            if ( !v33 )
            {
LABEL_65:
              ++*(_DWORD *)(a1 + 584);
              BUG();
            }
            v34 = v33 - 1;
            v7 = *(_QWORD *)(a1 + 576);
            v19 = v40;
            v35 = v34 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
            v28 = (_QWORD *)(v7 + 8LL * v35);
            v36 = *v28;
            v30 = *(_DWORD *)(a1 + 584) + 1;
            if ( *v28 != *v18 )
            {
              v41 = 1;
              v8 = 0;
              while ( v36 != -4096 )
              {
                if ( !v8 && v36 == -8192 )
                  v8 = (__int64)v28;
                v35 = v34 & (v41 + v35);
                v28 = (_QWORD *)(v7 + 8LL * v35);
                v36 = *v28;
                if ( *v18 == *v28 )
                  goto LABEL_40;
                ++v41;
              }
LABEL_30:
              if ( v8 )
                v28 = (_QWORD *)v8;
            }
          }
LABEL_40:
          *(_DWORD *)(a1 + 584) = v30;
          if ( *v28 != -4096 )
            --*(_DWORD *)(a1 + 588);
          *v28 = *v18;
          goto LABEL_23;
        }
LABEL_26:
        v37 = v19;
        sub_32B3220(v42, 2 * v24);
        v25 = *(_DWORD *)(a1 + 592);
        if ( !v25 )
          goto LABEL_65;
        v26 = v25 - 1;
        v7 = *(_QWORD *)(a1 + 576);
        v19 = v37;
        v27 = v26 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
        v28 = (_QWORD *)(v7 + 8LL * v27);
        v29 = *v28;
        v30 = *(_DWORD *)(a1 + 584) + 1;
        if ( *v28 != *v18 )
        {
          v38 = 1;
          v8 = 0;
          while ( v29 != -4096 )
          {
            if ( v29 == -8192 && !v8 )
              v8 = (__int64)v28;
            v27 = v26 & (v38 + v27);
            v28 = (_QWORD *)(v7 + 8LL * v27);
            v29 = *v28;
            if ( *v18 == *v28 )
              goto LABEL_40;
            ++v38;
          }
          goto LABEL_30;
        }
        goto LABEL_40;
      }
LABEL_23:
      if ( v20 == ++v18 )
      {
        v9 = v19;
        goto LABEL_4;
      }
    }
    ++*(_QWORD *)(a1 + 568);
    goto LABEL_26;
  }
LABEL_15:
  if ( *(_DWORD *)(v45 + 24) != 328 )
  {
    v47[0] = v45;
    sub_32B3B20(a1 + 568, v47);
    if ( *(int *)(v45 + 88) < 0 )
    {
      *(_DWORD *)(v45 + 88) = *(_DWORD *)(a1 + 48);
      v32 = *(unsigned int *)(a1 + 48);
      if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v32 + 1, 8u, v14, v15);
        v32 = *(unsigned int *)(a1 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v32) = v45;
      ++*(_DWORD *)(a1 + 48);
    }
  }
  return sub_32CF870(a1, a2[2]);
}
