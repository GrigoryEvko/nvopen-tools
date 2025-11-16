// Function: sub_136AFD0
// Address: 0x136afd0
//
char __fastcall sub_136AFD0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  int v7; // edi
  __int64 v8; // rcx
  int v9; // edx
  unsigned int v10; // r9d
  __int64 v11; // rax
  int v12; // r10d
  __int64 v13; // rdx
  unsigned int *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  _DWORD *v18; // rdi
  __int64 v19; // r14
  __int64 i; // r15
  __int64 v21; // r15
  unsigned int j; // r15d
  __int64 v23; // rax
  int v24; // ecx
  int v25; // edx
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // esi
  __int64 *v29; // rdx
  __int64 v30; // r10
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // edx
  int v34; // eax
  int v35; // r8d
  int v36; // r11d
  __int64 v38; // [rsp+10h] [rbp-50h]
  int v39; // [rsp+1Ch] [rbp-44h]
  int v40[13]; // [rsp+2Ch] [rbp-34h] BYREF

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
      goto LABEL_26;
    v9 = v31 - 1;
  }
  v10 = v9 & (37 * *a2);
  v11 = v8 + 16LL * v10;
  v12 = *(_DWORD *)v11;
  if ( *a2 == *(_DWORD *)v11 )
    goto LABEL_4;
  v34 = 1;
  while ( v12 != -1 )
  {
    v36 = v34 + 1;
    v10 = v9 & (v34 + v10);
    v11 = v8 + 16LL * v10;
    v12 = *(_DWORD *)v11;
    if ( *a2 == *(_DWORD *)v11 )
      goto LABEL_4;
    v34 = v36;
  }
  if ( (_BYTE)v7 )
  {
    v32 = 64;
    goto LABEL_27;
  }
  v31 = *(unsigned int *)(a1 + 72);
LABEL_26:
  v32 = 16 * v31;
LABEL_27:
  v11 = v8 + v32;
LABEL_4:
  v13 = 64;
  if ( !(_BYTE)v7 )
    v13 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v11 == v8 + v13 )
    return v11;
  v14 = *(unsigned int **)(v11 + 8);
  v15 = *(_QWORD *)(*(_QWORD *)a1 + 64LL) + 24LL * (unsigned int)*a2;
  v16 = *(_QWORD *)(v15 + 8);
  if ( v16 )
  {
    v17 = *(unsigned int *)(v16 + 12);
    v18 = *(_DWORD **)(v16 + 96);
    if ( (unsigned int)v17 > 1 )
    {
      LOBYTE(v11) = sub_1369030(v18, &v18[v17], (_DWORD *)v15);
      if ( (_BYTE)v11 )
      {
LABEL_10:
        if ( *(_BYTE *)(v16 + 8) )
        {
          v19 = *(_QWORD *)(v16 + 16);
          for ( i = v19 + 16LL * *(unsigned int *)(v16 + 24); i != v19; v19 += 16 )
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
  v21 = *(_QWORD *)(*(_QWORD *)(a4 + 136) + 8LL * *v14);
  v11 = sub_157EBA0(v21);
  if ( v11 )
  {
    v39 = sub_15F4D60(v11);
    v38 = sub_157EBA0(v21);
    LOBYTE(v11) = v39;
    if ( v39 )
    {
      for ( j = 0; j != v39; ++j )
      {
        v23 = sub_15F4DF0(v38, j);
        v24 = *(_DWORD *)(a4 + 184);
        v25 = -1;
        if ( v24 )
        {
          v26 = v24 - 1;
          v27 = *(_QWORD *)(a4 + 168);
          v28 = v26 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( v23 == *v29 )
          {
LABEL_20:
            v25 = *((_DWORD *)v29 + 2);
          }
          else
          {
            v33 = 1;
            while ( v30 != -8 )
            {
              v35 = v33 + 1;
              v28 = v26 & (v33 + v28);
              v29 = (__int64 *)(v27 + 16LL * v28);
              v30 = *v29;
              if ( v23 == *v29 )
                goto LABEL_20;
              v33 = v35;
            }
            v25 = -1;
          }
        }
        v40[0] = v25;
        LOBYTE(v11) = sub_1372900(a1, v14, v40, a3);
      }
    }
  }
  return v11;
}
