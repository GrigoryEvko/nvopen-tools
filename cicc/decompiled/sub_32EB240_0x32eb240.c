// Function: sub_32EB240
// Address: 0x32eb240
//
__int64 __fastcall sub_32EB240(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  int v5; // edx
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  _DWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 *v17; // r13
  __int64 v18; // r11
  __int64 *v19; // r14
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  __int64 v22; // rcx
  unsigned int v23; // esi
  int v24; // ecx
  int v25; // ecx
  unsigned int v26; // edx
  _QWORD *v27; // r10
  __int64 v28; // rdi
  int v29; // eax
  __int64 v30; // rax
  int v31; // eax
  int v32; // ecx
  int v33; // ecx
  unsigned int v34; // edx
  __int64 v35; // rdi
  __int64 v36; // [rsp+8h] [rbp-98h]
  int v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  int v39; // [rsp+8h] [rbp-98h]
  int v40; // [rsp+8h] [rbp-98h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+18h] [rbp-88h]
  const void *v43; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+38h] [rbp-68h] BYREF
  _BYTE v46[96]; // [rsp+40h] [rbp-60h] BYREF

  sub_325F8B0(a1, a2);
  v3 = *(_QWORD *)(a2 + 40);
  v4 = v3 + 40LL * *(unsigned int *)(a2 + 64);
  v43 = (const void *)(a1 + 56);
  v42 = a1 + 40;
  while ( v4 != v3 )
  {
    while ( 1 )
    {
      v10 = *(_DWORD **)v3;
      v11 = *(_QWORD *)(*(_QWORD *)v3 + 56LL);
      if ( (!v11 || *(_QWORD *)(v11 + 32)) && v10[17] <= 1u || v10[6] == 328 )
        goto LABEL_8;
      v5 = *(_DWORD *)(a1 + 584);
      v45 = *(_QWORD *)v3;
      if ( !v5 )
        break;
      sub_32B33F0((__int64)v46, a1 + 568, &v45);
      if ( !v46[32] )
        goto LABEL_7;
      v12 = *(unsigned int *)(a1 + 608);
      v13 = v45;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
      {
        sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v12 + 1, 8u, v8, v9);
        v12 = *(unsigned int *)(a1 + 608);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8 * v12) = v13;
      ++*(_DWORD *)(a1 + 608);
      if ( (int)v10[22] >= 0 )
        goto LABEL_8;
LABEL_16:
      v10[22] = *(_DWORD *)(a1 + 48);
      v14 = *(unsigned int *)(a1 + 48);
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(v42, v43, v14 + 1, 8u, v8, v9);
        v14 = *(unsigned int *)(a1 + 48);
      }
      v3 += 40;
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v14) = v10;
      ++*(_DWORD *)(a1 + 48);
      if ( v4 == v3 )
        return sub_33EBEB0(*(_QWORD *)a1, a2);
    }
    v6 = *(_QWORD **)(a1 + 600);
    v7 = &v6[*(unsigned int *)(a1 + 608)];
    if ( v7 == sub_325EB50(v6, (__int64)v7, &v45) )
    {
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
      {
        sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v8 + 1, 8u, v8, v9);
        v7 = (_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 608));
      }
      *v7 = v10;
      v16 = (unsigned int)(*(_DWORD *)(a1 + 608) + 1);
      *(_DWORD *)(a1 + 608) = v16;
      if ( (unsigned int)v16 > 0x20 )
      {
        v17 = *(__int64 **)(a1 + 600);
        v41 = a1 + 568;
        v18 = v4;
        v19 = &v17[v16];
        while ( 1 )
        {
          v23 = *(_DWORD *)(a1 + 592);
          if ( !v23 )
            break;
          v9 = v23 - 1;
          v8 = *(_QWORD *)(a1 + 576);
          v20 = v9 & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
          v21 = (_QWORD *)(v8 + 8LL * v20);
          v22 = *v21;
          if ( *v17 == *v21 )
          {
LABEL_25:
            if ( v19 == ++v17 )
              goto LABEL_33;
          }
          else
          {
            v37 = 1;
            v27 = 0;
            while ( v22 != -4096 )
            {
              if ( v22 != -8192 || v27 )
                v21 = v27;
              v20 = v9 & (v37 + v20);
              v22 = *(_QWORD *)(v8 + 8LL * v20);
              if ( *v17 == v22 )
                goto LABEL_25;
              ++v37;
              v27 = v21;
              v21 = (_QWORD *)(v8 + 8LL * v20);
            }
            v31 = *(_DWORD *)(a1 + 584);
            if ( !v27 )
              v27 = v21;
            ++*(_QWORD *)(a1 + 568);
            v29 = v31 + 1;
            if ( 4 * v29 < 3 * v23 )
            {
              if ( v23 - *(_DWORD *)(a1 + 588) - v29 > v23 >> 3 )
                goto LABEL_30;
              v38 = v18;
              sub_32B3220(v41, v23);
              v32 = *(_DWORD *)(a1 + 592);
              if ( !v32 )
              {
LABEL_64:
                ++*(_DWORD *)(a1 + 584);
                BUG();
              }
              v33 = v32 - 1;
              v8 = *(_QWORD *)(a1 + 576);
              v18 = v38;
              v34 = v33 & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
              v27 = (_QWORD *)(v8 + 8LL * v34);
              v35 = *v27;
              v29 = *(_DWORD *)(a1 + 584) + 1;
              if ( *v27 == *v17 )
                goto LABEL_30;
              v39 = 1;
              v9 = 0;
              while ( v35 != -4096 )
              {
                if ( !v9 && v35 == -8192 )
                  v9 = (__int64)v27;
                v34 = v33 & (v39 + v34);
                v27 = (_QWORD *)(v8 + 8LL * v34);
                v35 = *v27;
                if ( *v17 == *v27 )
                  goto LABEL_30;
                ++v39;
              }
              goto LABEL_51;
            }
LABEL_28:
            v36 = v18;
            sub_32B3220(v41, 2 * v23);
            v24 = *(_DWORD *)(a1 + 592);
            if ( !v24 )
              goto LABEL_64;
            v25 = v24 - 1;
            v8 = *(_QWORD *)(a1 + 576);
            v18 = v36;
            v26 = v25 & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
            v27 = (_QWORD *)(v8 + 8LL * v26);
            v28 = *v27;
            v29 = *(_DWORD *)(a1 + 584) + 1;
            if ( *v27 == *v17 )
              goto LABEL_30;
            v40 = 1;
            v9 = 0;
            while ( v28 != -4096 )
            {
              if ( v28 == -8192 && !v9 )
                v9 = (__int64)v27;
              v26 = v25 & (v40 + v26);
              v27 = (_QWORD *)(v8 + 8LL * v26);
              v28 = *v27;
              if ( *v17 == *v27 )
                goto LABEL_30;
              ++v40;
            }
LABEL_51:
            if ( v9 )
              v27 = (_QWORD *)v9;
LABEL_30:
            *(_DWORD *)(a1 + 584) = v29;
            if ( *v27 != -4096 )
              --*(_DWORD *)(a1 + 588);
            v30 = *v17++;
            *v27 = v30;
            if ( v19 == v17 )
            {
LABEL_33:
              v4 = v18;
              goto LABEL_7;
            }
          }
        }
        ++*(_QWORD *)(a1 + 568);
        goto LABEL_28;
      }
    }
LABEL_7:
    if ( (int)v10[22] < 0 )
      goto LABEL_16;
LABEL_8:
    v3 += 40;
  }
  return sub_33EBEB0(*(_QWORD *)a1, a2);
}
