// Function: sub_E84100
// Address: 0xe84100
//
__int64 __fastcall sub_E84100(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  const void *v7; // r15
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rdi
  char **v14; // rbx
  unsigned __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rdi
  _BYTE *v19; // rdi
  __int64 result; // rax
  _BYTE *v21; // rdi
  __int64 v22; // r8
  char *v23; // rbx
  int v24; // [rsp+0h] [rbp-60h] BYREF
  _BYTE *v25; // [rsp+8h] [rbp-58h] BYREF
  __int64 v26; // [rsp+10h] [rbp-50h]
  _BYTE dest[72]; // [rsp+18h] [rbp-48h] BYREF

  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(const void **)a3;
  v24 = a2;
  v8 = *(_QWORD *)(a1 + 296);
  v25 = dest;
  v9 = 8 * v6;
  v10 = *(_QWORD *)(v8 + 24);
  v26 = 0x300000000LL;
  if ( v6 > 3 )
  {
    sub_C8D5F0((__int64)&v25, dest, v6, 8u, a5, a6);
    v21 = &v25[8 * (unsigned int)v26];
  }
  else
  {
    if ( !v9 )
      goto LABEL_3;
    v21 = dest;
  }
  memcpy(v21, v7, 8 * v6);
  LODWORD(v9) = v26;
LABEL_3:
  v11 = *(unsigned int *)(v10 + 408);
  v12 = *(unsigned int *)(v10 + 412);
  LODWORD(v26) = v9 + v6;
  v13 = *(_QWORD *)(v10 + 400);
  v14 = (char **)&v24;
  v15 = v11 + 1;
  v16 = v11;
  if ( v11 + 1 > v12 )
  {
    v22 = v10 + 400;
    if ( v13 > (unsigned __int64)&v24 || (unsigned __int64)&v24 >= v13 + 48 * v11 )
    {
      sub_E83FF0(v10 + 400, v15, v11, v12, v22, a6);
      v11 = *(unsigned int *)(v10 + 408);
      v13 = *(_QWORD *)(v10 + 400);
      v16 = *(_DWORD *)(v10 + 408);
    }
    else
    {
      v23 = (char *)&v24 - v13;
      sub_E83FF0(v10 + 400, v15, v11, v12, v22, a6);
      v13 = *(_QWORD *)(v10 + 400);
      v11 = *(unsigned int *)(v10 + 408);
      v14 = (char **)&v23[v13];
      v16 = *(_DWORD *)(v10 + 408);
    }
  }
  v17 = 48 * v11;
  v18 = v17 + v13;
  if ( v18 )
  {
    *(_DWORD *)v18 = *(_DWORD *)v14;
    *(_QWORD *)(v18 + 8) = v18 + 24;
    *(_QWORD *)(v18 + 16) = 0x300000000LL;
    if ( *((_DWORD *)v14 + 4) )
    {
      v15 = (unsigned __int64)(v14 + 1);
      sub_E83220(v18 + 8, v14 + 1, v17, v12, a5, a6);
    }
    v16 = *(_DWORD *)(v10 + 408);
  }
  v19 = v25;
  result = (unsigned int)(v16 + 1);
  *(_DWORD *)(v10 + 408) = result;
  if ( v19 != dest )
    return _libc_free(v19, v15);
  return result;
}
