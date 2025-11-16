// Function: sub_E388C0
// Address: 0xe388c0
//
void *__fastcall sub_E388C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r13d
  __int64 v9; // rax
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rax
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v14; // r8
  unsigned __int64 v15; // r11
  _QWORD *v16; // r10
  int v17; // eax
  _QWORD *v18; // rdi
  _QWORD *v19; // rsi
  bool v20; // al
  __int64 *v21; // rsi
  _QWORD *v22; // rax
  __int64 i; // rdx
  __int64 v24; // rax
  const void *v25; // r12
  __int64 v26; // r13
  size_t v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  void *result; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 *v33; // rdi
  unsigned int v34; // edx
  __int64 *v35; // rsi
  __int64 v36; // rax
  int v37; // esi
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // rdx
  const void *v43; // rsi
  _DWORD *v44; // rsi
  __int64 v45; // [rsp+8h] [rbp-68h]
  unsigned __int64 v46; // [rsp+10h] [rbp-60h]
  int v47; // [rsp+18h] [rbp-58h]
  int v48; // [rsp+1Ch] [rbp-54h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+28h] [rbp-48h]
  __int64 v51; // [rsp+30h] [rbp-40h] BYREF
  __int64 v52[7]; // [rsp+38h] [rbp-38h] BYREF

  v8 = *(_DWORD *)(a1 + 184);
  if ( v8 )
  {
    result = (void *)(a1 + 176);
    if ( a2 != a1 + 176 )
    {
      v38 = *(unsigned int *)(a2 + 8);
      v39 = v8;
      if ( v8 <= v38 )
      {
        result = memmove(*(void **)a2, *(const void **)(a1 + 176), 8LL * v8);
        *(_DWORD *)(a2 + 8) = v8;
      }
      else
      {
        if ( v8 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v44 = (_DWORD *)(a2 + 16);
          v40 = 0;
          *(v44 - 2) = 0;
          sub_C8D5F0(a2, v44, v8, 8u, a5, a6);
          v39 = *(unsigned int *)(a1 + 184);
        }
        else
        {
          v40 = 8 * v38;
          if ( *(_DWORD *)(a2 + 8) )
          {
            memmove(*(void **)a2, *(const void **)(a1 + 176), 8 * v38);
            v39 = *(unsigned int *)(a1 + 184);
          }
        }
        v41 = *(_QWORD *)(a1 + 176);
        v42 = 8 * v39;
        v43 = (const void *)(v41 + v40);
        result = (void *)(v42 + v41);
        if ( v43 != result )
          result = memcpy((void *)(v40 + *(_QWORD *)a2), v43, v42 - v40);
        *(_DWORD *)(a2 + 8) = v8;
      }
    }
    return result;
  }
  *(_DWORD *)(a2 + 8) = 0;
  v9 = *(_QWORD *)(a1 + 88);
  v45 = v9 + 8LL * *(unsigned int *)(a1 + 96);
  if ( v45 == v9 )
  {
    v25 = *(const void **)a2;
    v28 = *(unsigned int *)(a1 + 184);
    v26 = 0;
    if ( *(_DWORD *)(a1 + 188) >= (unsigned int)v28 )
      goto LABEL_29;
    v29 = *(unsigned int *)(a1 + 184);
    v26 = 0;
    v27 = 0;
    goto LABEL_44;
  }
  v50 = *(_QWORD *)(a1 + 88);
  v10 = 0;
  do
  {
    v11 = *(_QWORD *)(*(_QWORD *)v50 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == *(_QWORD *)v50 + 48LL )
      goto LABEL_38;
    if ( !v11 )
      BUG();
    v12 = v11 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
    {
LABEL_38:
      v13 = 0;
      v14 = 0;
      v12 = 0;
    }
    else
    {
      v13 = sub_B46E30(v12);
      v14 = v12;
    }
    v49 &= 0xFFFFFFFF00000000LL;
    v46 = v13 | v46 & 0xFFFFFFFF00000000LL;
    sub_E37DE0((_DWORD *)a2, (__int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)), v12, v49, v14, v46);
    v15 = *(unsigned int *)(a2 + 8);
    if ( v15 > v10 )
    {
      a5 = v10;
      while ( 1 )
      {
        while ( 1 )
        {
          v16 = *(_QWORD **)a2;
          v17 = *(_DWORD *)(a1 + 72);
          a6 = *(_QWORD *)(*(_QWORD *)a2 + 8 * a5);
          v51 = a6;
          v52[0] = a6;
          if ( !v17 )
          {
            v18 = *(_QWORD **)(a1 + 88);
            v19 = &v18[*(unsigned int *)(a1 + 96)];
            v20 = v19 != sub_E37C60(v18, (__int64)v19, v52);
            goto LABEL_13;
          }
          v31 = *(unsigned int *)(a1 + 80);
          v32 = *(_QWORD *)(a1 + 64);
          v33 = (__int64 *)(v32 + 8 * v31);
          if ( (_DWORD)v31 )
            break;
LABEL_14:
          v21 = &v16[v10];
          if ( v21 != sub_E37D20(v16, (__int64)v21, &v51) )
            goto LABEL_10;
          ++a5;
          *v21 = a6;
          ++v10;
          if ( v15 == a5 )
          {
LABEL_16:
            v15 = *(unsigned int *)(a2 + 8);
            if ( v10 != v15 )
            {
              if ( v10 >= v15 )
                goto LABEL_18;
LABEL_24:
              *(_DWORD *)(a2 + 8) = v10;
            }
            goto LABEL_25;
          }
        }
        v48 = v31 - 1;
        v34 = (v31 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
        v35 = (__int64 *)(v32 + 8LL * ((*(_DWORD *)(a1 + 80) - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))));
        v36 = *v35;
        if ( a6 != *v35 )
        {
          v37 = 1;
          while ( v36 != -4096 )
          {
            v34 = v48 & (v37 + v34);
            v47 = v37 + 1;
            v35 = (__int64 *)(v32 + 8LL * v34);
            v36 = *v35;
            if ( a6 == *v35 )
              goto LABEL_33;
            v37 = v47;
          }
          goto LABEL_14;
        }
LABEL_33:
        v20 = v35 != v33;
LABEL_13:
        if ( !v20 )
          goto LABEL_14;
LABEL_10:
        if ( v15 == ++a5 )
          goto LABEL_16;
      }
    }
    if ( v15 != v10 )
    {
LABEL_18:
      if ( *(unsigned int *)(a2 + 12) < v10 )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v10, 8u, a5, a6);
        v15 = *(unsigned int *)(a2 + 8);
      }
      v22 = (_QWORD *)(*(_QWORD *)a2 + 8 * v15);
      for ( i = *(_QWORD *)a2 + 8 * v10; (_QWORD *)i != v22; ++v22 )
      {
        if ( v22 )
          *v22 = 0;
      }
      goto LABEL_24;
    }
LABEL_25:
    v50 += 8;
  }
  while ( v45 != v50 );
  v24 = *(unsigned int *)(a2 + 8);
  v25 = *(const void **)a2;
  v26 = v24;
  v27 = 8 * v24;
  v28 = *(unsigned int *)(a1 + 184);
  v29 = v28 + v26;
  if ( v28 + v26 <= (unsigned __int64)*(unsigned int *)(a1 + 188) )
  {
    if ( v27 )
      goto LABEL_28;
    goto LABEL_29;
  }
LABEL_44:
  sub_C8D5F0(a1 + 176, (const void *)(a1 + 192), v29, 8u, a5, a6);
  v28 = *(unsigned int *)(a1 + 184);
  if ( v27 )
  {
LABEL_28:
    memcpy((void *)(*(_QWORD *)(a1 + 176) + 8 * v28), v25, v27);
    v28 = *(unsigned int *)(a1 + 184);
  }
LABEL_29:
  result = (void *)(v26 + v28);
  *(_DWORD *)(a1 + 184) = (_DWORD)result;
  return result;
}
