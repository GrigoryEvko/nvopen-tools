// Function: sub_39CEC10
// Address: 0x39cec10
//
_QWORD *__fastcall sub_39CEC10(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rbx
  __int64 v8; // rdi
  unsigned int v9; // ecx
  _QWORD *result; // rax
  __int64 v11; // r10
  int v12; // eax
  int v13; // edx
  __int64 v14; // rsi
  unsigned int v15; // eax
  int v16; // ecx
  _QWORD *v17; // r15
  __int64 v18; // rdi
  int v19; // r9d
  _QWORD *v20; // r8
  int v21; // r11d
  int v22; // eax
  __int64 v23; // r14
  __int64 v24; // rbx
  char v25; // al
  __int64 v26; // r8
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned __int64 *v31; // r8
  unsigned __int64 v32; // r9
  __int64 v33; // rax
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  int v37; // r9d
  unsigned int v38; // edx
  __int64 v39; // rsi
  unsigned __int8 *v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rax
  int v43; // edx
  int v44; // esi
  __int64 v45; // rdi
  unsigned int v46; // ecx
  __int64 *v47; // rdx
  __int64 v48; // r9
  int v49; // edx
  int v50; // r10d
  unsigned int v51; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v52; // [rsp+8h] [rbp-48h]
  _BYTE v53[52]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = (_QWORD *)a1;
  v4 = *(_QWORD *)(a2 + 8);
  if ( !sub_39C7370(a1) || (v7 = a1 + 864, (unsigned __int8)sub_3989C80(*(_QWORD *)(a1 + 200))) )
  {
    v5 = *(_QWORD *)(a1 + 208);
    v6 = *(_DWORD *)(v5 + 320);
    v7 = v5 + 296;
    if ( !v6 )
    {
LABEL_8:
      ++*(_QWORD *)v7;
      goto LABEL_9;
    }
  }
  else
  {
    v6 = *(_DWORD *)(a1 + 888);
    if ( !v6 )
      goto LABEL_8;
  }
  v8 = *(_QWORD *)(v7 + 8);
  v9 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  result = (_QWORD *)(v8 + 16LL * v9);
  v11 = *result;
  if ( v4 != *result )
  {
    v21 = 1;
    v17 = 0;
    while ( v11 != -8 )
    {
      if ( !v17 && v11 == -16 )
        v17 = result;
      v9 = (v6 - 1) & (v21 + v9);
      result = (_QWORD *)(v8 + 16LL * v9);
      v11 = *result;
      if ( v4 == *result )
        goto LABEL_4;
      ++v21;
    }
    if ( !v17 )
      v17 = result;
    v22 = *(_DWORD *)(v7 + 16);
    ++*(_QWORD *)v7;
    v16 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v7 + 20) - v16 > v6 >> 3 )
        goto LABEL_22;
      v51 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
      sub_39A53F0(v7, v6);
      v34 = *(_DWORD *)(v7 + 24);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(v7 + 8);
        v37 = 1;
        v20 = 0;
        v38 = v35 & v51;
        v16 = *(_DWORD *)(v7 + 16) + 1;
        v17 = (_QWORD *)(v36 + 16LL * (v35 & v51));
        v39 = *v17;
        if ( v4 != *v17 )
        {
          while ( v39 != -8 )
          {
            if ( v39 == -16 && !v20 )
              v20 = v17;
            v38 = v35 & (v37 + v38);
            v17 = (_QWORD *)(v36 + 16LL * v38);
            v39 = *v17;
            if ( v4 == *v17 )
              goto LABEL_22;
            ++v37;
          }
          goto LABEL_13;
        }
LABEL_22:
        *(_DWORD *)(v7 + 16) = v16;
        if ( *v17 != -8 )
          --*(_DWORD *)(v7 + 20);
        *v17 = v4;
        v17[1] = 0;
        goto LABEL_25;
      }
LABEL_59:
      ++*(_DWORD *)(v7 + 16);
      BUG();
    }
LABEL_9:
    sub_39A53F0(v7, 2 * v6);
    v12 = *(_DWORD *)(v7 + 24);
    if ( v12 )
    {
      v13 = v12 - 1;
      v14 = *(_QWORD *)(v7 + 8);
      v15 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v16 = *(_DWORD *)(v7 + 16) + 1;
      v17 = (_QWORD *)(v14 + 16LL * v15);
      v18 = *v17;
      if ( v4 != *v17 )
      {
        v19 = 1;
        v20 = 0;
        while ( v18 != -8 )
        {
          if ( !v20 && v18 == -16 )
            v20 = v17;
          v15 = v13 & (v19 + v15);
          v17 = (_QWORD *)(v14 + 16LL * v15);
          v18 = *v17;
          if ( v4 == *v17 )
            goto LABEL_22;
          ++v19;
        }
LABEL_13:
        if ( v20 )
          v17 = v20;
        goto LABEL_22;
      }
      goto LABEL_22;
    }
    goto LABEL_59;
  }
LABEL_4:
  if ( result[1] )
    return result;
  v17 = result;
LABEL_25:
  v23 = *(_QWORD *)(a2 + 8);
  v24 = (__int64)v3;
  v25 = sub_39C84F0(v3);
  v26 = (__int64)(v3 + 1);
  if ( !v25 )
  {
    v27 = *(unsigned int *)(v23 + 8);
    v28 = *(_QWORD *)(v23 + 8 * (6 - v27));
    if ( !v28 )
    {
      v40 = sub_39A81B0((__int64)v3, *(unsigned __int8 **)(v23 + 8 * (1 - v27)));
      v41 = v3[25];
      v52 = v40;
      v42 = sub_3981E80((__int64)v40);
      v43 = *(_DWORD *)(v41 + 600);
      v26 = (__int64)v52;
      if ( v43 )
      {
        v44 = v43 - 1;
        v45 = *(_QWORD *)(v41 + 584);
        v46 = (v43 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v47 = (__int64 *)(v45 + 16LL * v46);
        v48 = *v47;
        if ( v42 == *v47 )
        {
LABEL_42:
          v3 = (_QWORD *)v47[1];
          v24 = (__int64)v3;
          goto LABEL_28;
        }
        v49 = 1;
        while ( v48 != -8 )
        {
          v50 = v49 + 1;
          v46 = v44 & (v49 + v46);
          v47 = (__int64 *)(v45 + 16LL * v46);
          v48 = *v47;
          if ( v42 == *v47 )
            goto LABEL_42;
          v49 = v50;
        }
      }
      v24 = 0;
      v3 = 0;
      goto LABEL_28;
    }
    sub_39A8220((__int64)v3, v28, 0);
    v26 = (__int64)(v3 + 1);
  }
LABEL_28:
  v29 = sub_39A5A90(v24, 46, v26, 0);
  v17[1] = v29;
  sub_39C9540(v3, v23, v29);
  if ( !sub_39C84F0(v3) )
  {
    v33 = v17[1];
    v53[2] = 0;
    sub_39A3560(v24, (__int64 *)(v33 + 8), 32, (__int64)v53, 1);
  }
  result = (_QWORD *)sub_39CEB00((__int64)v3, a2, v17[1], v30, v31, v32);
  if ( result )
    return (_QWORD *)sub_39A3B20(v24, v17[1], 100, (__int64)result);
  return result;
}
