// Function: sub_32B3670
// Address: 0x32b3670
//
__int64 __fastcall sub_32B3670(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r15
  _QWORD *v5; // rdi
  _QWORD *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // r13
  _QWORD *v15; // r14
  __int64 v16; // r11
  unsigned int v17; // eax
  _QWORD *v18; // rdi
  __int64 v19; // rcx
  unsigned int v20; // esi
  int v21; // ecx
  int v22; // ecx
  unsigned int v23; // edx
  _QWORD *v24; // r10
  __int64 v25; // rdi
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  int v29; // ecx
  unsigned int v30; // edx
  __int64 v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-88h]
  int v33; // [rsp+8h] [rbp-88h]
  int v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+8h] [rbp-88h]
  int v36; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  const void *v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v40[96]; // [rsp+30h] [rbp-60h] BYREF

  v38 = (const void *)(a1 + 56);
  result = a1 + 40;
  v37 = a1 + 40;
  if ( a2 )
  {
    v4 = a2;
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v4 + 16);
        if ( *(_DWORD *)(v9 + 24) == 328 )
          goto LABEL_5;
        v10 = *(_DWORD *)(a1 + 584);
        v39 = *(_QWORD *)(v4 + 16);
        if ( !v10 )
          break;
        sub_32B33F0((__int64)v40, a1 + 568, &v39);
        if ( !v40[32] )
          goto LABEL_4;
        v11 = *(unsigned int *)(a1 + 608);
        v12 = v39;
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
        {
          sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v11 + 1, 8u, v7, v8);
          v11 = *(unsigned int *)(a1 + 608);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8 * v11) = v12;
        ++*(_DWORD *)(a1 + 608);
        result = *(unsigned int *)(v9 + 88);
        if ( (int)result < 0 )
          goto LABEL_12;
LABEL_5:
        v4 = *(_QWORD *)(v4 + 32);
        if ( !v4 )
          return result;
      }
      v5 = *(_QWORD **)(a1 + 600);
      v6 = &v5[*(unsigned int *)(a1 + 608)];
      if ( v6 == sub_325EB50(v5, (__int64)v6, &v39) )
      {
        if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
        {
          sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v7 + 1, 8u, v7, v8);
          v6 = (_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 608));
        }
        *v6 = v9;
        v13 = (unsigned int)(*(_DWORD *)(a1 + 608) + 1);
        *(_DWORD *)(a1 + 608) = v13;
        if ( (unsigned int)v13 > 0x20 )
          break;
      }
LABEL_4:
      result = *(unsigned int *)(v9 + 88);
      if ( (int)result >= 0 )
        goto LABEL_5;
LABEL_12:
      *(_DWORD *)(v9 + 88) = *(_DWORD *)(a1 + 48);
      result = *(unsigned int *)(a1 + 48);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(v37, v38, result + 1, 8u, v7, v8);
        result = *(unsigned int *)(a1 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * result) = v9;
      ++*(_DWORD *)(a1 + 48);
      v4 = *(_QWORD *)(v4 + 32);
      if ( !v4 )
        return result;
    }
    v14 = *(_QWORD **)(a1 + 600);
    v15 = &v14[v13];
    v16 = a1 + 568;
    while ( 1 )
    {
      v20 = *(_DWORD *)(a1 + 592);
      if ( !v20 )
        break;
      v8 = v20 - 1;
      v7 = *(_QWORD *)(a1 + 576);
      v17 = v8 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v18 = (_QWORD *)(v7 + 8LL * v17);
      v19 = *v18;
      if ( *v14 != *v18 )
      {
        v34 = 1;
        v24 = 0;
        while ( v19 != -4096 )
        {
          if ( v24 || v19 != -8192 )
            v18 = v24;
          v17 = v8 & (v34 + v17);
          v19 = *(_QWORD *)(v7 + 8LL * v17);
          if ( *v14 == v19 )
            goto LABEL_21;
          ++v34;
          v24 = v18;
          v18 = (_QWORD *)(v7 + 8LL * v17);
        }
        v27 = *(_DWORD *)(a1 + 584);
        if ( !v24 )
          v24 = v18;
        ++*(_QWORD *)(a1 + 568);
        v26 = v27 + 1;
        if ( 4 * v26 < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a1 + 588) - v26 <= v20 >> 3 )
          {
            v35 = v16;
            sub_32B3220(v16, v20);
            v28 = *(_DWORD *)(a1 + 592);
            if ( !v28 )
            {
LABEL_59:
              ++*(_DWORD *)(a1 + 584);
              BUG();
            }
            v29 = v28 - 1;
            v7 = *(_QWORD *)(a1 + 576);
            v16 = v35;
            v30 = v29 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
            v24 = (_QWORD *)(v7 + 8LL * v30);
            v31 = *v24;
            v26 = *(_DWORD *)(a1 + 584) + 1;
            if ( *v24 != *v14 )
            {
              v36 = 1;
              v8 = 0;
              while ( v31 != -4096 )
              {
                if ( !v8 && v31 == -8192 )
                  v8 = (__int64)v24;
                v30 = v29 & (v36 + v30);
                v24 = (_QWORD *)(v7 + 8LL * v30);
                v31 = *v24;
                if ( *v14 == *v24 )
                  goto LABEL_37;
                ++v36;
              }
LABEL_28:
              if ( v8 )
                v24 = (_QWORD *)v8;
            }
          }
LABEL_37:
          *(_DWORD *)(a1 + 584) = v26;
          if ( *v24 != -4096 )
            --*(_DWORD *)(a1 + 588);
          *v24 = *v14;
          goto LABEL_21;
        }
LABEL_24:
        v32 = v16;
        sub_32B3220(v16, 2 * v20);
        v21 = *(_DWORD *)(a1 + 592);
        if ( !v21 )
          goto LABEL_59;
        v22 = v21 - 1;
        v7 = *(_QWORD *)(a1 + 576);
        v16 = v32;
        v23 = v22 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
        v24 = (_QWORD *)(v7 + 8LL * v23);
        v25 = *v24;
        v26 = *(_DWORD *)(a1 + 584) + 1;
        if ( *v24 != *v14 )
        {
          v33 = 1;
          v8 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v8 )
              v8 = (__int64)v24;
            v23 = v22 & (v33 + v23);
            v24 = (_QWORD *)(v7 + 8LL * v23);
            v25 = *v24;
            if ( *v14 == *v24 )
              goto LABEL_37;
            ++v33;
          }
          goto LABEL_28;
        }
        goto LABEL_37;
      }
LABEL_21:
      if ( v15 == ++v14 )
        goto LABEL_4;
    }
    ++*(_QWORD *)(a1 + 568);
    goto LABEL_24;
  }
  return result;
}
