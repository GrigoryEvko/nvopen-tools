// Function: sub_20C35E0
// Address: 0x20c35e0
//
unsigned int *__fastcall sub_20C35E0(__int64 a1, unsigned int a2, int a3)
{
  __int16 v3; // cx
  _QWORD *v4; // rdx
  __int64 *v5; // r12
  __int64 v6; // rax
  __int64 v8; // r9
  unsigned int v9; // edi
  _WORD *v10; // r8
  unsigned __int16 v11; // si
  _WORD *v12; // rdi
  unsigned __int16 v13; // r8
  __int64 v14; // r11
  __int64 v15; // rsi
  __int16 *v16; // rsi
  __int16 v17; // ax
  __int16 *v18; // rsi
  unsigned __int16 v19; // cx
  __int16 *v20; // rax
  __int16 v21; // dx
  __int64 v22; // rax
  unsigned int *result; // rax
  __int64 *v24; // rax
  __int64 v25; // rdx
  unsigned __int16 *v26; // rsi
  _WORD *v27; // rax
  __int16 v28; // dx
  unsigned __int16 *v29; // rax
  unsigned __int16 v30; // r13
  unsigned __int16 *v31; // r15
  __int64 v32; // rsi
  __int16 v33; // si
  unsigned int v35; // [rsp+Ch] [rbp-74h] BYREF
  unsigned int v36; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v37; // [rsp+18h] [rbp-68h]
  char v38; // [rsp+20h] [rbp-60h]
  unsigned __int16 v39; // [rsp+28h] [rbp-58h]
  _WORD *v40; // [rsp+30h] [rbp-50h]
  int v41; // [rsp+38h] [rbp-48h]
  unsigned __int16 v42; // [rsp+40h] [rbp-40h]
  __int64 v43; // [rsp+48h] [rbp-38h]

  v3 = a2;
  v4 = *(_QWORD **)(a1 + 32);
  v35 = a2;
  if ( !v4 )
  {
    v36 = a2;
    v37 = 0;
    v38 = 1;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    BUG();
  }
  v36 = a2;
  v37 = v4 + 1;
  v5 = *(__int64 **)(a1 + 72);
  v6 = a2;
  v39 = 0;
  v8 = a2;
  v40 = 0;
  v38 = 1;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v9 = *(_DWORD *)(v4[1] + 24LL * a2 + 16);
  v10 = (_WORD *)(v4[7] + 2LL * (v9 >> 4));
  v11 = *v10 + a2 * (v9 & 0xF);
  v12 = v10 + 1;
  v39 = v11;
  v40 = v10 + 1;
  while ( 1 )
  {
    if ( !v12 )
      goto LABEL_34;
    v41 = *(_DWORD *)(v4[6] + 4LL * v39);
    v13 = v41;
    if ( (_WORD)v41 )
      break;
LABEL_32:
    v40 = ++v12;
    v33 = *(v12 - 1);
    v39 += v33;
    if ( !v33 )
    {
      v40 = 0;
LABEL_34:
      v24 = v5;
LABEL_18:
      if ( *(_DWORD *)(v24[13] + 4 * v8) == -1 || (result = (unsigned int *)v24[16], result[v8] != -1) )
      {
        *(_DWORD *)(v5[13] + 4 * v8) = a3;
        *(_DWORD *)(v5[16] + 4LL * v35) = -1;
        sub_20C3510(v5 + 7, &v35);
        sub_20C3180(*(_QWORD **)(a1 + 72), v35);
        v25 = *(_QWORD *)(a1 + 32);
        if ( !v25 )
          BUG();
        v26 = 0;
        v27 = (_WORD *)(*(_QWORD *)(v25 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v25 + 8) + 24LL * v35 + 4));
        v28 = *v27;
        v29 = v27 + 1;
        v30 = v28 + v35;
        if ( v28 )
          v26 = v29;
        result = &v36;
        while ( 1 )
        {
          v31 = v26;
          if ( !v26 )
            break;
          while ( 1 )
          {
            v32 = *(_QWORD *)(a1 + 72);
            v36 = v30;
            if ( *(_DWORD *)(*(_QWORD *)(v32 + 104) + 4LL * v30) == -1
              || *(_DWORD *)(*(_QWORD *)(v32 + 128) + 4LL * v30) != -1 )
            {
              *(_DWORD *)(v5[13] + 4LL * v30) = a3;
              *(_DWORD *)(v5[16] + 4LL * v36) = -1;
              sub_20C3510(v5 + 7, &v36);
              sub_20C3180(*(_QWORD **)(a1 + 72), v36);
            }
            result = (unsigned int *)*v31;
            v26 = 0;
            ++v31;
            if ( !(_WORD)result )
              break;
            v30 += (unsigned __int16)result;
            if ( !v31 )
              return result;
          }
        }
      }
      return result;
    }
  }
  while ( 1 )
  {
    v14 = *(unsigned int *)(v4[1] + 24LL * v13 + 8);
    v15 = v4[7];
    v42 = v13;
    v43 = v15 + 2 * v14;
    if ( v43 )
      break;
    v13 = HIWORD(v41);
    v41 = HIWORD(v41);
    if ( !v13 )
      goto LABEL_32;
  }
  while ( 1 )
  {
    v16 = (__int16 *)(v4[7] + 2LL * *(unsigned int *)(v4[1] + 24 * v6 + 8));
    v17 = *v16;
    v18 = v16 + 1;
    v19 = v17 + v3;
    if ( !v17 )
      v18 = 0;
LABEL_8:
    v20 = v18;
    if ( v18 )
    {
      while ( v19 != v13 )
      {
        v21 = *v20;
        v18 = 0;
        ++v20;
        if ( !v21 )
          goto LABEL_8;
        v19 += v21;
        if ( !v20 )
          goto LABEL_12;
      }
      v22 = *(_QWORD *)(a1 + 72);
      if ( *(_DWORD *)(*(_QWORD *)(v22 + 104) + 4LL * v19) != -1 )
      {
        result = *(unsigned int **)(v22 + 128);
        if ( result[v19] == -1 )
          return result;
      }
    }
LABEL_12:
    sub_1E1D5E0((__int64)&v36);
    if ( !v40 )
    {
      v24 = *(__int64 **)(a1 + 72);
      v8 = v35;
      goto LABEL_18;
    }
    v6 = v35;
    v4 = *(_QWORD **)(a1 + 32);
    v13 = v42;
    v3 = v35;
  }
}
