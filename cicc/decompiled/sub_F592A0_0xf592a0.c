// Function: sub_F592A0
// Address: 0xf592a0
//
__int64 __fastcall sub_F592A0(__int64 a1, __int64 **a2, __int64 **a3)
{
  __int64 v5; // r12
  int v6; // r13d
  __int64 *v7; // rsi
  __int64 v8; // rax
  __int64 *v9; // rdi
  int v10; // eax
  __int64 v11; // r14
  int v12; // r9d
  unsigned int v13; // r15d
  __int64 *v14; // r8
  bool v15; // dl
  __int64 *v16; // rcx
  __int64 v17; // rsi
  char v18; // al
  __int64 result; // rax
  unsigned int v20; // r15d
  int v21; // [rsp+14h] [rbp-3Ch]
  __int64 *v22; // [rsp+18h] [rbp-38h]

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
    v9 = (__int64 *)*(v7 - 1);
    v7 = &v9[(unsigned __int64)v8 / 8];
  }
  else
  {
    v9 = &v7[v8 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v10 = sub_F58E90(v9, v7);
  v11 = (__int64)*a2;
  v12 = 1;
  v13 = v6 & v10;
  v14 = 0;
  v15 = v11 == -4096 || v11 == -8192;
  while ( 1 )
  {
    v16 = (__int64 *)(v5 + 8LL * v13);
    v17 = *v16;
    if ( *v16 == -8192 || *v16 == -4096 || v15 )
      break;
    v21 = v12;
    v22 = v14;
    v18 = sub_B46220(v11, v17);
    v14 = v22;
    v12 = v21;
    v16 = (__int64 *)(v5 + 8LL * v13);
    v15 = 0;
    if ( v18 )
      goto LABEL_9;
LABEL_21:
    v20 = v12 + v13;
    ++v12;
    v13 = v6 & v20;
  }
  if ( v17 == v11 )
  {
LABEL_9:
    *a3 = v16;
    return 1;
  }
  if ( v17 != -4096 )
  {
    if ( !v14 && *v16 == -8192 )
      v14 = (__int64 *)(v5 + 8LL * v13);
    goto LABEL_21;
  }
  if ( !v14 )
    v14 = (__int64 *)(v5 + 8LL * v13);
  *a3 = v14;
  return 0;
}
