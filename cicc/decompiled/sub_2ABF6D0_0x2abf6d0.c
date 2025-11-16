// Function: sub_2ABF6D0
// Address: 0x2abf6d0
//
__int64 __fastcall sub_2ABF6D0(__int64 a1, __int64 **a2, __int64 **a3)
{
  __int64 v5; // r13
  int v6; // r14d
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rsi
  __int64 *v10; // rdi
  int v11; // eax
  __int64 v12; // rdi
  int v13; // r8d
  unsigned int v14; // edx
  __int64 *v15; // rcx
  bool v16; // r15
  __int64 *v17; // rbx
  __int64 v18; // rsi
  char v19; // al
  __int64 result; // rax
  unsigned int v21; // edx
  int v22; // [rsp+8h] [rbp-58h]
  unsigned int v23; // [rsp+Ch] [rbp-54h]
  __int64 *v24; // [rsp+10h] [rbp-50h]
  int v25; // [rsp+24h] [rbp-3Ch] BYREF
  unsigned __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = result - 1;
    if ( !(_DWORD)result )
    {
      *a3 = 0;
      return result;
    }
  }
  v7 = *a2;
  v8 = 32LL * (*((_DWORD *)*a2 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)*a2 + 7) & 0x40) != 0 )
  {
    v10 = (__int64 *)*(v7 - 1);
    v9 = &v10[(unsigned __int64)v8 / 8];
  }
  else
  {
    v9 = *a2;
    v10 = &v7[v8 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v26[0] = sub_2ABF340(v10, v9);
  v25 = *(unsigned __int8 *)v7 - 29;
  v11 = sub_C4ECF0(&v25, (__int64 *)v26);
  v12 = (__int64)*a2;
  v13 = 1;
  v14 = v6 & v11;
  v15 = 0;
  v16 = *a2 + 1024 == 0 || *a2 + 512 == 0;
  while ( 1 )
  {
    v17 = (__int64 *)(v5 + 16LL * v14);
    v18 = *v17;
    if ( *v17 == -8192 || *v17 == -4096 || v16 )
      break;
    v22 = v13;
    v23 = v14;
    v24 = v15;
    v19 = sub_B46220(v12, v18);
    v15 = v24;
    v14 = v23;
    v13 = v22;
    if ( v19 )
      goto LABEL_9;
LABEL_21:
    v21 = v13 + v14;
    ++v13;
    v14 = v6 & v21;
  }
  if ( v18 == v12 )
  {
LABEL_9:
    *a3 = v17;
    return 1;
  }
  if ( v18 != -4096 )
  {
    if ( !v15 && *v17 == -8192 )
      v15 = (__int64 *)(v5 + 16LL * v14);
    goto LABEL_21;
  }
  if ( !v15 )
    v15 = (__int64 *)(v5 + 16LL * v14);
  *a3 = v15;
  return 0;
}
