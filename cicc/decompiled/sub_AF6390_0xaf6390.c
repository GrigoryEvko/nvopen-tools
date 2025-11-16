// Function: sub_AF6390
// Address: 0xaf6390
//
__int64 __fastcall sub_AF6390(__int64 a1, __int64 a2, char a3)
{
  unsigned __int64 *v4; // r14
  unsigned __int64 *v5; // rbx
  _BYTE *v6; // rax
  unsigned __int64 *v7; // rsi
  __int64 v8; // rax
  unsigned __int64 *v9; // r14
  __int64 result; // rax
  char *v11; // r14
  char *v12; // r10
  char *v13; // r14
  size_t v14; // r11
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  size_t v19; // [rsp+0h] [rbp-60h]
  char *v21; // [rsp+8h] [rbp-58h]
  char v22; // [rsp+17h] [rbp-49h]
  unsigned __int64 *v23; // [rsp+18h] [rbp-48h]
  void *src; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v25[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(unsigned __int64 **)(a2 + 16);
  v5 = *(unsigned __int64 **)(a2 + 24);
  v22 = a3;
  v25[0] = v4;
  if ( v5 == v4 )
  {
LABEL_20:
    v17 = *(unsigned int *)(a1 + 8);
    if ( v17 + 2 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, a1 + 16, v17 + 2, 8);
      v17 = *(unsigned int *)(a1 + 8);
    }
    v18 = *(_QWORD *)a1;
    *(_QWORD *)(v18 + 8 * v17) = 4101;
    *(_QWORD *)(v18 + 8 * v17 + 8) = 0;
    *(_DWORD *)(a1 + 8) += 2;
  }
  else
  {
    while ( *v4 != 4101 )
    {
      v4 += (unsigned int)sub_AF4160(v25);
      v25[0] = v4;
      if ( v5 == v4 )
        goto LABEL_20;
    }
  }
  v6 = *(_BYTE **)(a2 + 24);
  v7 = *(unsigned __int64 **)(a2 + 16);
  v23 = *(unsigned __int64 **)(a2 + 24);
  if ( !a3 )
    return sub_AF5AE0(a1, v7, v6);
  v25[0] = v7;
  if ( v23 == v7 )
    return sub_A188E0(a1, 6);
  do
  {
    src = v25[0];
    if ( *v7 == 159 || *v7 == 4096 )
    {
      v16 = *(unsigned int *)(a1 + 8);
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, a1 + 16, v16 + 1, 8);
        v16 = *(unsigned int *)(a1 + 8);
      }
      v22 = 0;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v16) = 6;
      ++*(_DWORD *)(a1 + 8);
    }
    v11 = (char *)src;
    v12 = &v11[8 * (unsigned int)sub_AF4160((unsigned __int64 **)&src)];
    v13 = (char *)src;
    v8 = *(unsigned int *)(a1 + 8);
    v14 = v12 - (_BYTE *)src;
    v15 = (v12 - (_BYTE *)src) >> 3;
    if ( v15 + v8 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v19 = v12 - (_BYTE *)src;
      v21 = v12;
      sub_C8D5F0(a1, a1 + 16, v15 + v8, 8);
      v8 = *(unsigned int *)(a1 + 8);
      v14 = v19;
      v12 = v21;
    }
    if ( v12 != v13 )
    {
      memcpy((void *)(*(_QWORD *)a1 + 8 * v8), v13, v14);
      LODWORD(v8) = *(_DWORD *)(a1 + 8);
    }
    v9 = v25[0];
    *(_DWORD *)(a1 + 8) = v15 + v8;
    result = (unsigned int)sub_AF4160(v25);
    v7 = &v9[(unsigned int)result];
    v25[0] = v7;
  }
  while ( v23 != v7 );
  if ( v22 )
    return sub_A188E0(a1, 6);
  return result;
}
