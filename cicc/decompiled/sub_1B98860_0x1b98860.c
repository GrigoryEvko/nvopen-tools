// Function: sub_1B98860
// Address: 0x1b98860
//
__int64 __fastcall sub_1B98860(__int64 a1, __int64 **a2, __int64 **a3)
{
  __int64 v5; // r14
  int v6; // r15d
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rsi
  __int64 *v10; // rdi
  int v11; // eax
  int v12; // r8d
  __int64 *v13; // rcx
  unsigned int i; // edx
  __int64 v15; // rdi
  __int64 *v16; // rbx
  __int64 v17; // rsi
  char v18; // al
  __int64 result; // rax
  unsigned int v20; // edx
  int v21; // [rsp+0h] [rbp-50h]
  unsigned int v22; // [rsp+4h] [rbp-4Ch]
  __int64 *v23; // [rsp+8h] [rbp-48h]
  int v24; // [rsp+14h] [rbp-3Ch] BYREF
  unsigned __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

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
  v8 = 24LL * (*((_DWORD *)*a2 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)*a2 + 23) & 0x40) != 0 )
  {
    v10 = (__int64 *)*(v7 - 1);
    v9 = &v10[(unsigned __int64)v8 / 8];
  }
  else
  {
    v9 = *a2;
    v10 = &v7[v8 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v25[0] = sub_1B98460(v10, v9);
  v24 = *((unsigned __int8 *)v7 + 16) - 24;
  v11 = sub_18FDAA0(&v24, (__int64 *)v25);
  v12 = 1;
  v13 = 0;
  for ( i = v6 & v11; ; i = v6 & v20 )
  {
    v15 = (__int64)*a2;
    v16 = (__int64 *)(v5 + 16LL * i);
    v17 = *v16;
    if ( *v16 == -8 || *a2 + 2 == 0 || *a2 + 1 == 0 || v17 == -16 )
    {
      if ( v17 == v15 )
      {
LABEL_9:
        *a3 = v16;
        return 1;
      }
    }
    else
    {
      v21 = v12;
      v22 = i;
      v23 = v13;
      v18 = sub_15F41F0(v15, v17);
      v13 = v23;
      i = v22;
      v12 = v21;
      if ( v18 )
        goto LABEL_9;
      v17 = *v16;
    }
    if ( v17 == -8 )
      break;
    if ( v17 == -16 && !v13 )
      v13 = v16;
    v20 = v12 + i;
    ++v12;
  }
  if ( !v13 )
    v13 = v16;
  *a3 = v13;
  return 0;
}
