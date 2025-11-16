// Function: sub_31E4280
// Address: 0x31e4280
//
__int64 __fastcall sub_31E4280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdi
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r10d
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  unsigned int v14; // edx
  __int64 *v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  _QWORD *v19; // rdi
  int v20; // eax
  __int64 *v21; // rdx
  __int64 result; // rax
  int v23; // eax
  int v24; // edx
  _QWORD *v25; // rax
  int v26; // eax
  int v27; // ecx
  __int64 v28; // r8
  unsigned int v29; // eax
  __int64 v30; // rsi
  int v31; // r10d
  _QWORD *v32; // r9
  int v33; // eax
  int v34; // eax
  __int64 v35; // rsi
  int v36; // r9d
  _QWORD *v37; // r8
  unsigned int v38; // r15d
  __int64 v39; // rcx
  const char *v40; // [rsp+0h] [rbp-60h] BYREF
  char v41; // [rsp+20h] [rbp-40h]
  char v42; // [rsp+21h] [rbp-3Fh]

  v7 = *(_QWORD *)(a2 + 24);
  v42 = 1;
  v40 = "pcsection";
  v41 = 3;
  v8 = sub_E6C380(v7, (__int64 *)&v40, 1, a4, a5);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v8, 0);
  v9 = *(_DWORD *)(a1 + 528);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 504);
    goto LABEL_24;
  }
  v10 = v9 - 1;
  v11 = 1;
  v12 = *(_QWORD *)(a1 + 512);
  v13 = 0;
  v14 = v10 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v12 + 72LL * v14);
  v16 = *v15;
  if ( a3 != *v15 )
  {
    while ( v16 != -4096 )
    {
      if ( !v13 && v16 == -8192 )
        v13 = v15;
      v14 = v10 & (v11 + v14);
      v15 = (__int64 *)(v12 + 72LL * v14);
      v16 = *v15;
      if ( a3 == *v15 )
        goto LABEL_3;
      ++v11;
    }
    v23 = *(_DWORD *)(a1 + 520);
    if ( !v13 )
      v13 = v15;
    ++*(_QWORD *)(a1 + 504);
    v24 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 524) - v24 > v9 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 520) = v24;
        if ( *v13 != -4096 )
          --*(_DWORD *)(a1 + 524);
        v25 = v13 + 3;
        *v13 = a3;
        v17 = 0;
        v19 = v13 + 1;
        *v19 = v25;
        v19[1] = 0x600000000LL;
        v20 = 0;
        goto LABEL_4;
      }
      sub_31E3F60(a1 + 504, v9);
      v33 = *(_DWORD *)(a1 + 528);
      if ( v33 )
      {
        v34 = v33 - 1;
        v35 = *(_QWORD *)(a1 + 512);
        v36 = 1;
        v37 = 0;
        v38 = v34 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v13 = (_QWORD *)(v35 + 72LL * v38);
        v39 = *v13;
        v24 = *(_DWORD *)(a1 + 520) + 1;
        if ( a3 != *v13 )
        {
          while ( v39 != -4096 )
          {
            if ( !v37 && v39 == -8192 )
              v37 = v13;
            v38 = v34 & (v36 + v38);
            v13 = (_QWORD *)(v35 + 72LL * v38);
            v39 = *v13;
            if ( a3 == *v13 )
              goto LABEL_20;
            ++v36;
          }
          if ( v37 )
            v13 = v37;
        }
        goto LABEL_20;
      }
LABEL_47:
      ++*(_DWORD *)(a1 + 520);
      BUG();
    }
LABEL_24:
    sub_31E3F60(a1 + 504, 2 * v9);
    v26 = *(_DWORD *)(a1 + 528);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 512);
      v29 = (v26 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v13 = (_QWORD *)(v28 + 72LL * v29);
      v30 = *v13;
      v24 = *(_DWORD *)(a1 + 520) + 1;
      if ( a3 != *v13 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( !v32 && v30 == -8192 )
            v32 = v13;
          v29 = v27 & (v31 + v29);
          v13 = (_QWORD *)(v28 + 72LL * v29);
          v30 = *v13;
          if ( a3 == *v13 )
            goto LABEL_20;
          ++v31;
        }
        if ( v32 )
          v13 = v32;
      }
      goto LABEL_20;
    }
    goto LABEL_47;
  }
LABEL_3:
  v17 = *((unsigned int *)v15 + 4);
  v18 = *((unsigned int *)v15 + 5);
  v19 = v15 + 1;
  v20 = *((_DWORD *)v15 + 4);
  if ( v18 > v17 )
  {
LABEL_4:
    v21 = (__int64 *)(*v19 + 8 * v17);
    if ( v21 )
    {
      *v21 = v8;
      v20 = *((_DWORD *)v19 + 2);
    }
    result = (unsigned int)(v20 + 1);
    *((_DWORD *)v19 + 2) = result;
    return result;
  }
  if ( v18 < v17 + 1 )
  {
    sub_C8D5F0((__int64)v19, v15 + 3, v17 + 1, 8u, v17 + 1, v10);
    v17 = *((unsigned int *)v15 + 4);
  }
  result = v15[1];
  *(_QWORD *)(result + 8 * v17) = v8;
  ++*((_DWORD *)v15 + 4);
  return result;
}
