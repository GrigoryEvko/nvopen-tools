// Function: sub_1DDEB40
// Address: 0x1ddeb40
//
char __fastcall sub_1DDEB40(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  int v7; // edi
  __int64 v8; // rcx
  int v9; // edx
  unsigned int v10; // r10d
  __int64 v11; // rax
  int v12; // r11d
  __int64 v13; // rdx
  __int64 *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  _DWORD *v18; // rdi
  _DWORD *v19; // rbx
  _DWORD *i; // r14
  _QWORD *v21; // rbx
  int v22; // edx
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // r11
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // eax
  int v32; // eax
  int v33; // r10d
  int v34; // ebx
  _QWORD *j; // [rsp+0h] [rbp-50h]
  __int64 v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+8h] [rbp-48h]
  int v39[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v7 = *(_BYTE *)(a1 + 56) & 1;
  if ( v7 )
  {
    v8 = a1 + 64;
    v9 = 3;
  }
  else
  {
    v29 = *(unsigned int *)(a1 + 72);
    v8 = *(_QWORD *)(a1 + 64);
    if ( !(_DWORD)v29 )
      goto LABEL_24;
    v9 = v29 - 1;
  }
  v10 = v9 & (37 * *a2);
  v11 = v8 + 16LL * v10;
  v12 = *(_DWORD *)v11;
  if ( *a2 == *(_DWORD *)v11 )
    goto LABEL_4;
  v32 = 1;
  while ( v12 != -1 )
  {
    v34 = v32 + 1;
    v10 = v9 & (v32 + v10);
    v11 = v8 + 16LL * v10;
    v12 = *(_DWORD *)v11;
    if ( *a2 == *(_DWORD *)v11 )
      goto LABEL_4;
    v32 = v34;
  }
  if ( (_BYTE)v7 )
  {
    v30 = 64;
    goto LABEL_25;
  }
  v29 = *(unsigned int *)(a1 + 72);
LABEL_24:
  v30 = 16 * v29;
LABEL_25:
  v11 = v8 + v30;
LABEL_4:
  v13 = 64;
  if ( !(_BYTE)v7 )
    v13 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v11 == v8 + v13 )
    return v11;
  v14 = *(__int64 **)(v11 + 8);
  v15 = *(_QWORD *)(*(_QWORD *)a1 + 64LL) + 24LL * (unsigned int)*a2;
  v16 = *(_QWORD *)(v15 + 8);
  if ( v16 )
  {
    v17 = *(unsigned int *)(v16 + 12);
    v18 = *(_DWORD **)(v16 + 96);
    if ( (unsigned int)v17 > 1 )
    {
      v37 = a4;
      LOBYTE(v11) = sub_1369030(v18, &v18[v17], (_DWORD *)v15);
      a4 = v37;
      if ( (_BYTE)v11 )
      {
LABEL_10:
        if ( *(_BYTE *)(v16 + 8) )
        {
          v19 = *(_DWORD **)(v16 + 16);
          for ( i = &v19[4 * *(unsigned int *)(v16 + 24)]; i != v19; v19 += 4 )
            LOBYTE(v11) = sub_1372900(a1, v14, v19, a3);
          return v11;
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
  v11 = *(_QWORD *)(*(_QWORD *)(a4 + 136) + 8LL * *(unsigned int *)v14);
  v21 = *(_QWORD **)(v11 + 88);
  for ( j = *(_QWORD **)(v11 + 96); j != v21; a4 = v38 )
  {
    v22 = *(_DWORD *)(a4 + 184);
    v23 = -1;
    if ( v22 )
    {
      v24 = v22 - 1;
      v25 = *(_QWORD *)(a4 + 168);
      v26 = v24 & (((unsigned int)*v21 >> 9) ^ ((unsigned int)*v21 >> 4));
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( *v21 == *v27 )
      {
LABEL_18:
        v23 = *((_DWORD *)v27 + 2);
      }
      else
      {
        v31 = 1;
        while ( v28 != -8 )
        {
          v33 = v31 + 1;
          v26 = v24 & (v31 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( *v21 == *v27 )
            goto LABEL_18;
          v31 = v33;
        }
        v23 = -1;
      }
    }
    v38 = a4;
    ++v21;
    v39[0] = v23;
    LOBYTE(v11) = sub_1372900(a1, v14, v39, a3);
  }
  return v11;
}
