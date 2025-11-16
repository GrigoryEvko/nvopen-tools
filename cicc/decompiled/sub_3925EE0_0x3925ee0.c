// Function: sub_3925EE0
// Address: 0x3925ee0
//
__int64 __fastcall sub_3925EE0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // r11d
  _QWORD *v12; // r14
  int v13; // eax
  int v14; // edx
  unsigned __int64 *v15; // rsi
  unsigned __int64 v16; // rdx
  const void *v17; // rsi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r9d
  _QWORD *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  unsigned int v29; // r13d
  _QWORD *v30; // rdi
  _BYTE *v31; // rcx

  v3 = a1 + 192;
  v5 = *(_DWORD *)(a1 + 216);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 192);
    goto LABEL_19;
  }
  v6 = *(_QWORD *)(a1 + 200);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( (_BYTE *)*v8 != a2 )
  {
    v11 = 1;
    v12 = 0;
    while ( v9 != -8 )
    {
      if ( !v12 && v9 == -16 )
        v12 = v8;
      v7 = (v5 - 1) & (v11 + v7);
      v8 = (_QWORD *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( (_BYTE *)*v8 == a2 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v8;
    v13 = *(_DWORD *)(a1 + 208);
    ++*(_QWORD *)(a1 + 192);
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 212) - v14 > v5 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 208) = v14;
        if ( *v12 != -8 )
          --*(_DWORD *)(a1 + 212);
        *v12 = a2;
        v12[1] = 0;
        goto LABEL_14;
      }
      sub_3925D30(v3, v5);
      v25 = *(_DWORD *)(a1 + 216);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 200);
        v28 = 1;
        v29 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v14 = *(_DWORD *)(a1 + 208) + 1;
        v30 = 0;
        v12 = (_QWORD *)(v27 + 16LL * v29);
        v31 = (_BYTE *)*v12;
        if ( (_BYTE *)*v12 != a2 )
        {
          while ( v31 != (_BYTE *)-8LL )
          {
            if ( !v30 && v31 == (_BYTE *)-16LL )
              v30 = v12;
            v29 = v26 & (v28 + v29);
            v12 = (_QWORD *)(v27 + 16LL * v29);
            v31 = (_BYTE *)*v12;
            if ( (_BYTE *)*v12 == a2 )
              goto LABEL_11;
            ++v28;
          }
          if ( v30 )
            v12 = v30;
        }
        goto LABEL_11;
      }
LABEL_48:
      ++*(_DWORD *)(a1 + 208);
      BUG();
    }
LABEL_19:
    sub_3925D30(v3, 2 * v5);
    v18 = *(_DWORD *)(a1 + 216);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 200);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 208) + 1;
      v12 = (_QWORD *)(v20 + 16LL * v21);
      v22 = *v12;
      if ( (_BYTE *)*v12 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v12;
          v21 = v19 & (v23 + v21);
          v12 = (_QWORD *)(v20 + 16LL * v21);
          v22 = *v12;
          if ( (_BYTE *)*v12 == a2 )
            goto LABEL_11;
          ++v23;
        }
        if ( v24 )
          v12 = v24;
      }
      goto LABEL_11;
    }
    goto LABEL_48;
  }
LABEL_3:
  if ( v8[1] )
    return v8[1];
  v12 = v8;
LABEL_14:
  if ( (*a2 & 4) != 0 )
  {
    v15 = (unsigned __int64 *)*((_QWORD *)a2 - 1);
    v16 = *v15;
    v17 = v15 + 2;
  }
  else
  {
    v16 = 0;
    v17 = 0;
  }
  result = sub_3925260((_QWORD *)a1, v17, v16);
  v12[1] = result;
  return result;
}
