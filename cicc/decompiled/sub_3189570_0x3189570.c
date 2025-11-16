// Function: sub_3189570
// Address: 0x3189570
//
_QWORD *__fastcall sub_3189570(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  _QWORD *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // r9
  int v8; // r11d
  unsigned __int64 v9; // rdx
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v15; // rax
  __int64 v16; // r9
  _QWORD *v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // rdi
  char *v20; // rcx
  unsigned __int64 v21; // rsi
  __int64 v22; // r8
  int v23; // eax
  int v24; // eax
  int v25; // ecx
  int v26; // eax
  int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // r10d
  _QWORD *v32; // r9
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  _QWORD *v36; // r8
  unsigned int v37; // r14d
  int v38; // r9d
  __int64 v39; // rsi
  __int64 v40; // rdi
  char *v41; // rbx
  _QWORD v42[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 88;
  v4 = *(_QWORD **)a2;
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 16LL);
  *(_QWORD *)a2 = 0;
  v6 = *(unsigned int *)(a1 + 112);
  if ( !(_DWORD)v6 )
  {
    ++*(_QWORD *)(a1 + 88);
    goto LABEL_28;
  }
  v7 = *(_QWORD *)(a1 + 96);
  v8 = 1;
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v11 = (__int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( v5 == *v11 )
  {
LABEL_3:
    (*(void (__fastcall **)(_QWORD *, __int64, unsigned __int64, __int64, __int64))(*v4 + 8LL))(v4, v6, v9, v12, v2);
    goto LABEL_4;
  }
  while ( v12 != -4096 )
  {
    if ( v9 || v12 != -8192 )
      v11 = (__int64 *)v9;
    v9 = (unsigned int)(v8 + 1);
    v10 = (v6 - 1) & (v8 + v10);
    v12 = *(_QWORD *)(v7 + 16LL * v10);
    if ( v5 == v12 )
      goto LABEL_3;
    ++v8;
    v9 = (unsigned __int64)v11;
    v11 = (__int64 *)(v7 + 16LL * v10);
  }
  if ( !v9 )
    v9 = (unsigned __int64)v11;
  v24 = *(_DWORD *)(a1 + 104);
  ++*(_QWORD *)(a1 + 88);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= (unsigned int)(3 * v6) )
  {
LABEL_28:
    sub_3187CD0(v2, 2 * v6);
    v26 = *(_DWORD *)(a1 + 112);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 96);
      v29 = (v26 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v25 = *(_DWORD *)(a1 + 104) + 1;
      v9 = v28 + 16LL * v29;
      v30 = *(_QWORD *)v9;
      if ( v5 != *(_QWORD *)v9 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( !v32 && v30 == -8192 )
            v32 = (_QWORD *)v9;
          v29 = v27 & (v31 + v29);
          v9 = v28 + 16LL * v29;
          v30 = *(_QWORD *)v9;
          if ( v5 == *(_QWORD *)v9 )
            goto LABEL_24;
          ++v31;
        }
        if ( v32 )
          v9 = (unsigned __int64)v32;
      }
      goto LABEL_24;
    }
    goto LABEL_56;
  }
  if ( (int)v6 - *(_DWORD *)(a1 + 108) - v25 <= (unsigned int)v6 >> 3 )
  {
    sub_3187CD0(v2, v6);
    v33 = *(_DWORD *)(a1 + 112);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 96);
      v36 = 0;
      v37 = v34 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v38 = 1;
      v25 = *(_DWORD *)(a1 + 104) + 1;
      v9 = v35 + 16LL * v37;
      v39 = *(_QWORD *)v9;
      if ( v5 != *(_QWORD *)v9 )
      {
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v36 )
            v36 = (_QWORD *)v9;
          v37 = v34 & (v38 + v37);
          v9 = v35 + 16LL * v37;
          v39 = *(_QWORD *)v9;
          if ( v5 == *(_QWORD *)v9 )
            goto LABEL_24;
          ++v38;
        }
        if ( v36 )
          v9 = (unsigned __int64)v36;
      }
      goto LABEL_24;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 104);
    BUG();
  }
LABEL_24:
  *(_DWORD *)(a1 + 104) = v25;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 108);
  *(_QWORD *)v9 = v5;
  *(_QWORD *)(v9 + 8) = v4;
LABEL_4:
  if ( (unsigned __int8)sub_318B630(v4) )
  {
    if ( *(_DWORD *)(a1 + 72) == 1 )
    {
      v15 = (_QWORD *)sub_22077B0(0x10u);
      v17 = v15;
      if ( v15 )
      {
        v15[1] = v4;
        *v15 = &unk_4A34720;
      }
      v18 = *(unsigned int *)(a1 + 16);
      v19 = *(unsigned int *)(a1 + 20);
      v42[0] = v15;
      v20 = (char *)v42;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = v18 + 1;
      v23 = v18;
      if ( v18 + 1 > v19 )
      {
        v40 = a1 + 8;
        if ( v21 > (unsigned __int64)v42 || (unsigned __int64)v42 >= v21 + 8 * v18 )
        {
          sub_31878D0(v40, v18 + 1, v18, (__int64)v42, v22, v16);
          v18 = *(unsigned int *)(a1 + 16);
          v21 = *(_QWORD *)(a1 + 8);
          v20 = (char *)v42;
          v23 = *(_DWORD *)(a1 + 16);
        }
        else
        {
          v41 = (char *)v42 - v21;
          sub_31878D0(v40, v18 + 1, v18, (__int64)v42 - v21, v22, v16);
          v21 = *(_QWORD *)(a1 + 8);
          v18 = *(unsigned int *)(a1 + 16);
          v20 = &v41[v21];
          v23 = *(_DWORD *)(a1 + 16);
        }
      }
      v13 = (_QWORD *)(v21 + 8 * v18);
      if ( v13 )
      {
        *v13 = *(_QWORD *)v20;
        *(_QWORD *)v20 = 0;
        v17 = (_QWORD *)v42[0];
        v23 = *(_DWORD *)(a1 + 16);
      }
      *(_DWORD *)(a1 + 16) = v23 + 1;
      if ( v17 )
        (*(void (__fastcall **)(_QWORD *, unsigned __int64, _QWORD *, char *))(*v17 + 24LL))(v17, v21, v13, v20);
    }
    sub_3187110(a1, v4, (__int64)v13);
  }
  return v4;
}
