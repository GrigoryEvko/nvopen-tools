// Function: sub_103E3E0
// Address: 0x103e3e0
//
__int64 *__fastcall sub_103E3E0(__int64 a1, unsigned __int8 *a2)
{
  int v3; // eax
  __int64 v5; // rsi
  int v6; // ecx
  unsigned int v7; // edx
  unsigned __int8 **v8; // rax
  unsigned __int8 *v9; // rdi
  int v10; // eax
  _QWORD *v11; // rax
  unsigned int v12; // ecx
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 *result; // rax
  __int64 v17; // r9
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  int v20; // eax
  int v21; // r10d
  int v22; // eax
  int v23; // r8d

  v3 = *(_DWORD *)(a1 + 320);
  v5 = *(_QWORD *)(a1 + 304);
  if ( v3 )
  {
    v6 = v3 - 1;
    v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (unsigned __int8 **)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      *v8 = (unsigned __int8 *)-8192LL;
      --*(_DWORD *)(a1 + 312);
      ++*(_DWORD *)(a1 + 316);
    }
    else
    {
      v22 = 1;
      while ( v9 != (unsigned __int8 *)-4096LL )
      {
        v23 = v22 + 1;
        v7 = v6 & (v22 + v7);
        v8 = (unsigned __int8 **)(v5 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v22 = v23;
      }
    }
  }
  v10 = *a2;
  if ( v10 == 26 )
  {
    v18 = a2 - 32;
  }
  else
  {
    if ( v10 != 27 )
      goto LABEL_6;
    v18 = a2 - 64;
  }
  if ( *(_QWORD *)v18 )
  {
    v19 = *((_QWORD *)v18 + 1);
    **((_QWORD **)v18 + 2) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *((_QWORD *)v18 + 2);
  }
  *(_QWORD *)v18 = 0;
  if ( *a2 == 26 )
  {
LABEL_16:
    v12 = *(_DWORD *)(a1 + 56);
    v13 = *((_QWORD *)a2 + 9);
    v14 = *(_QWORD *)(a1 + 40);
    if ( !v12 )
      goto LABEL_17;
    goto LABEL_8;
  }
LABEL_6:
  v11 = sub_103E0E0((_QWORD *)a1);
  (*(void (__fastcall **)(_QWORD *, unsigned __int8 *))(*v11 + 32LL))(v11, a2);
  if ( (unsigned int)*a2 - 26 <= 1 )
    goto LABEL_16;
  v12 = *(_DWORD *)(a1 + 56);
  v13 = *((_QWORD *)a2 + 8);
  v14 = *(_QWORD *)(a1 + 40);
  if ( !v12 )
    goto LABEL_17;
LABEL_8:
  v15 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  result = (__int64 *)(v14 + 16LL * v15);
  v17 = *result;
  if ( v13 == *result )
  {
LABEL_9:
    if ( (unsigned __int8 *)result[1] != a2 )
      return result;
LABEL_18:
    *result = -8192;
    --*(_DWORD *)(a1 + 48);
    ++*(_DWORD *)(a1 + 52);
    return result;
  }
  v20 = 1;
  while ( v17 != -4096 )
  {
    v21 = v20 + 1;
    v15 = (v12 - 1) & (v20 + v15);
    result = (__int64 *)(v14 + 16LL * v15);
    v17 = *result;
    if ( v13 == *result )
      goto LABEL_9;
    v20 = v21;
  }
LABEL_17:
  result = (__int64 *)(v14 + 16LL * v12);
  if ( (unsigned __int8 *)result[1] == a2 )
    goto LABEL_18;
  return result;
}
