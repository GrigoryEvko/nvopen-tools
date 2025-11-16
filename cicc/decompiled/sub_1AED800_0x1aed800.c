// Function: sub_1AED800
// Address: 0x1aed800
//
__int64 __fastcall sub_1AED800(__int64 a1, __int64 **a2, __int64 **a3)
{
  __int64 v5; // r12
  int v6; // r13d
  __int64 *v7; // rsi
  __int64 v8; // rax
  __int64 *v9; // rdi
  int v10; // eax
  int v11; // r8d
  __int64 *v12; // rcx
  unsigned int i; // r15d
  __int64 v14; // rdi
  __int64 *v15; // rdx
  __int64 v16; // rsi
  char v17; // al
  __int64 result; // rax
  unsigned int v19; // r15d
  int v20; // [rsp+14h] [rbp-3Ch]
  __int64 *v21; // [rsp+18h] [rbp-38h]

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
    v9 = (__int64 *)*(v7 - 1);
    v7 = &v9[(unsigned __int64)v8 / 8];
  }
  else
  {
    v9 = &v7[v8 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v10 = sub_18FDB50(v9, v7);
  v11 = 1;
  v12 = 0;
  for ( i = v6 & v10; ; i = v6 & v19 )
  {
    v14 = (__int64)*a2;
    v15 = (__int64 *)(v5 + 8LL * i);
    v16 = *v15;
    if ( *v15 == -8 || *a2 + 1 == 0 || *a2 + 2 == 0 || v16 == -16 )
    {
      if ( v16 == v14 )
      {
LABEL_9:
        *a3 = v15;
        return 1;
      }
    }
    else
    {
      v20 = v11;
      v21 = v12;
      v17 = sub_15F41F0(v14, v16);
      v12 = v21;
      v11 = v20;
      v15 = (__int64 *)(v5 + 8LL * i);
      if ( v17 )
        goto LABEL_9;
      v16 = *(_QWORD *)(v5 + 8LL * i);
    }
    if ( v16 == -8 )
      break;
    if ( v16 == -16 && !v12 )
      v12 = v15;
    v19 = v11 + i;
    ++v11;
  }
  if ( !v12 )
    v12 = v15;
  *a3 = v12;
  return 0;
}
