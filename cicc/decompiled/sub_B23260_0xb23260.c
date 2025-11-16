// Function: sub_B23260
// Address: 0xb23260
//
__int64 *__fastcall sub_B23260(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 *result; // rax
  __int64 v7; // r13
  __int64 *v8; // rsi
  __int64 *v9; // rdi
  __int64 *v10; // r15
  __int64 *v11; // rbx
  __int64 v12; // r8
  unsigned int v13; // edx
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rcx
  int v18; // edx
  __int64 *v19; // rdi
  __int64 *v20; // r9
  int v21; // r8d
  __int64 v22; // rax
  __int64 *v23; // rsi
  __int64 v24; // r11
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // r12
  int v28; // edx
  int v29; // r13d
  int v30; // r11d
  _QWORD *v31; // r10
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // [rsp+18h] [rbp-98h]
  __int64 v35; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v36; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v37; // [rsp+30h] [rbp-80h] BYREF
  int v38; // [rsp+38h] [rbp-78h]
  char v39; // [rsp+40h] [rbp-70h] BYREF

  v2 = *a1;
  if ( *(_BYTE *)(*a1 + 32) )
  {
    sub_C7D6A0(*(_QWORD *)(v2 + 8), 16LL * *(unsigned int *)(v2 + 24), 8);
    ++*(_QWORD *)v2;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 24) = 0;
  }
  else
  {
    *(_QWORD *)v2 = 1;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 24) = 0;
    *(_BYTE *)(v2 + 32) = 1;
  }
  sub_C7D6A0(0, 0, 8);
  v3 = *(_QWORD *)(a1[1] + 128);
  v4 = v3 + 72;
  v5 = *(_QWORD *)(v3 + 80);
  result = (__int64 *)&v36;
  v34 = v4;
  if ( v5 == v4 )
    return result;
LABEL_6:
  while ( 2 )
  {
    v7 = v5 - 24;
    if ( !v5 )
      v7 = 0;
    if ( !*(_DWORD *)sub_B20CA0(a1[2], v7) )
    {
      v8 = (__int64 *)v7;
      sub_B1CB80(&v37, v7, *(_QWORD *)(a1[2] + 4128));
      v9 = v37;
      v10 = &v37[v38];
      if ( v10 == v37 )
        goto LABEL_18;
      v11 = v37;
      while ( 1 )
      {
        v16 = *a1;
        v17 = *v11;
        v8 = (__int64 *)*(unsigned int *)(*a1 + 24);
        v35 = *v11;
        if ( !(_DWORD)v8 )
          break;
        v12 = *(_QWORD *)(v16 + 8);
        v13 = ((_DWORD)v8 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v14 = (_QWORD *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v17 != *v14 )
        {
          v30 = 1;
          v31 = 0;
          while ( v15 != -4096 )
          {
            if ( v31 || v15 != -8192 )
              v14 = v31;
            v13 = ((_DWORD)v8 - 1) & (v30 + v13);
            v15 = *(_QWORD *)(v12 + 16LL * v13);
            if ( v17 == v15 )
              goto LABEL_12;
            ++v30;
            v31 = v14;
            v14 = (_QWORD *)(v12 + 16LL * v13);
          }
          if ( !v31 )
            v31 = v14;
          v18 = *(_DWORD *)(v16 + 16) + 1;
          v36 = v31;
          ++*(_QWORD *)v16;
          if ( 4 * v18 < (unsigned int)(3 * (_DWORD)v8) )
          {
            if ( (int)v8 - *(_DWORD *)(v16 + 20) - v18 > (unsigned int)v8 >> 3 )
              goto LABEL_40;
            goto LABEL_16;
          }
LABEL_15:
          LODWORD(v8) = 2 * (_DWORD)v8;
LABEL_16:
          sub_B23080(v16, (int)v8);
          v8 = &v35;
          sub_B1C700(v16, &v35, &v36);
          v18 = *(_DWORD *)(v16 + 16) + 1;
LABEL_40:
          v32 = v36;
          *(_DWORD *)(v16 + 16) = v18;
          if ( *v32 != -4096 )
            --*(_DWORD *)(v16 + 20);
          v33 = v35;
          *((_DWORD *)v32 + 2) = 0;
          *v32 = v33;
        }
LABEL_12:
        if ( v10 == ++v11 )
        {
          v9 = v37;
LABEL_18:
          if ( v9 == (__int64 *)&v39 )
            goto LABEL_5;
          _libc_free(v9, v8);
          v5 = *(_QWORD *)(v5 + 8);
          if ( v34 == v5 )
            goto LABEL_20;
          goto LABEL_6;
        }
      }
      v36 = 0;
      ++*(_QWORD *)v16;
      goto LABEL_15;
    }
LABEL_5:
    v5 = *(_QWORD *)(v5 + 8);
    if ( v34 != v5 )
      continue;
    break;
  }
LABEL_20:
  result = *(__int64 **)(a1[1] + 128);
  v19 = (__int64 *)result[10];
  v20 = result + 9;
  if ( result + 9 != v19 )
  {
    v21 = 0;
    do
    {
      v22 = *a1;
      v23 = v19 - 3;
      if ( !v19 )
        v23 = 0;
      ++v21;
      v24 = *(_QWORD *)(v22 + 8);
      result = (__int64 *)*(unsigned int *)(v22 + 24);
      if ( (_DWORD)result )
      {
        v25 = ((_DWORD)result - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( v23 == (__int64 *)*v26 )
        {
LABEL_26:
          result = (__int64 *)(v24 + 16LL * (_QWORD)result);
          if ( v26 != result )
            *((_DWORD *)v26 + 2) = v21;
        }
        else
        {
          v28 = 1;
          while ( v27 != -4096 )
          {
            v29 = v28 + 1;
            v25 = ((_DWORD)result - 1) & (v28 + v25);
            v26 = (__int64 *)(v24 + 16LL * v25);
            v27 = *v26;
            if ( v23 == (__int64 *)*v26 )
              goto LABEL_26;
            v28 = v29;
          }
        }
      }
      v19 = (__int64 *)v19[1];
    }
    while ( v20 != v19 );
  }
  return result;
}
