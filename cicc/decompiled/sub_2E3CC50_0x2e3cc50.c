// Function: sub_2E3CC50
// Address: 0x2e3cc50
//
char __fastcall sub_2E3CC50(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  char v7; // cl
  __int64 v8; // rdi
  int v9; // edx
  unsigned int v10; // r10d
  _QWORD *v11; // rax
  int v12; // r11d
  __int64 v13; // rdx
  __int64 *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  _DWORD *v18; // rdi
  _DWORD *v19; // rbx
  _DWORD *i; // r14
  __int64 v21; // rax
  _QWORD *v22; // r9
  int v23; // ecx
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r11
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rax
  int v32; // eax
  int v33; // r10d
  __int64 v34; // rax
  int v35; // eax
  int v36; // r8d
  _QWORD *v38; // [rsp+0h] [rbp-50h]
  _QWORD *v39; // [rsp+8h] [rbp-48h]
  int v40[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v7 = *(_BYTE *)(a1 + 56) & 1;
  if ( v7 )
  {
    v8 = a1 + 64;
    v9 = 3;
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 72);
    v8 = *(_QWORD *)(a1 + 64);
    if ( !(_DWORD)v31 )
      goto LABEL_29;
    v9 = v31 - 1;
  }
  v10 = v9 & (37 * *a2);
  v11 = (_QWORD *)(v8 + 16LL * v10);
  v12 = *(_DWORD *)v11;
  if ( *(_DWORD *)v11 == *a2 )
    goto LABEL_4;
  v35 = 1;
  while ( v12 != -1 )
  {
    v36 = v35 + 1;
    v10 = v9 & (v35 + v10);
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = *(_DWORD *)v11;
    if ( *a2 == *(_DWORD *)v11 )
      goto LABEL_4;
    v35 = v36;
  }
  if ( v7 )
  {
    v34 = 64;
    goto LABEL_30;
  }
  v31 = *(unsigned int *)(a1 + 72);
LABEL_29:
  v34 = 16 * v31;
LABEL_30:
  v11 = (_QWORD *)(v8 + v34);
LABEL_4:
  v13 = 64;
  if ( !v7 )
    v13 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v11 != (_QWORD *)(v8 + v13) )
  {
    v14 = (__int64 *)v11[1];
    v15 = *(_QWORD *)(*(_QWORD *)a1 + 64LL) + 24LL * (unsigned int)*a2;
    v16 = *(_QWORD *)(v15 + 8);
    if ( v16 )
    {
      v17 = *(unsigned int *)(v16 + 12);
      v18 = *(_DWORD **)(v16 + 96);
      if ( (unsigned int)v17 > 1 )
      {
        LOBYTE(v11) = sub_FDC990(v18, &v18[v17], (_DWORD *)v15);
        if ( (_BYTE)v11 )
        {
LABEL_10:
          if ( *(_BYTE *)(v16 + 8) )
          {
            v19 = *(_DWORD **)(v16 + 16);
            for ( i = &v19[4 * *(unsigned int *)(v16 + 24)]; i != v19; v19 += 4 )
              LOBYTE(v11) = sub_FEB360(a1, v14, v19, a3);
            return (char)v11;
          }
        }
      }
      else
      {
        LODWORD(v11) = *v18;
        if ( *(_DWORD *)v15 == *v18 )
          goto LABEL_10;
      }
    }
    v21 = *(_QWORD *)(*(_QWORD *)(a4 + 136) + 8LL * *(unsigned int *)v14);
    v22 = *(_QWORD **)(v21 + 112);
    v11 = &v22[*(unsigned int *)(v21 + 120)];
    v38 = v11;
    if ( v22 == v11 )
      return (char)v11;
    while ( 1 )
    {
      v28 = *(_DWORD *)(a4 + 184);
      v29 = *v22;
      v30 = *(_QWORD *)(a4 + 168);
      if ( !v28 )
        goto LABEL_21;
      v23 = v28 - 1;
      v24 = v23 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v25 = (__int64 *)(v30 + 16LL * v24);
      v26 = *v25;
      if ( v29 != *v25 )
        break;
LABEL_18:
      v27 = *((_DWORD *)v25 + 2);
LABEL_19:
      v39 = v22;
      v40[0] = v27;
      LOBYTE(v11) = sub_FEB360(a1, v14, v40, a3);
      v22 = v39 + 1;
      if ( v38 == v39 + 1 )
        return (char)v11;
    }
    v32 = 1;
    while ( v26 != -4096 )
    {
      v33 = v32 + 1;
      v24 = v23 & (v32 + v24);
      v25 = (__int64 *)(v30 + 16LL * v24);
      v26 = *v25;
      if ( v29 == *v25 )
        goto LABEL_18;
      v32 = v33;
    }
LABEL_21:
    v27 = -1;
    goto LABEL_19;
  }
  return (char)v11;
}
