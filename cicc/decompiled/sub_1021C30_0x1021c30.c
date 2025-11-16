// Function: sub_1021C30
// Address: 0x1021c30
//
__int64 *__fastcall sub_1021C30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        char a9,
        char a10,
        __int64 a11,
        int a12)
{
  int v14; // r12d
  int v15; // ebx
  char v16; // dl
  char v17; // al
  bool v18; // zf
  __int64 *result; // rax
  __int64 v20; // rdx
  __int64 *v21; // r14
  __int64 v22; // rsi
  __int64 *v23; // r12
  char v24; // di
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // rdx

  v14 = a5;
  v15 = a6;
  v16 = a9;
  v17 = a10;
  *(_QWORD *)a1 = a4;
  *(_QWORD *)(a1 + 8) = 6;
  *(_QWORD *)(a1 + 16) = 0;
  if ( a2 )
  {
    *(_QWORD *)(a1 + 24) = a2;
    if ( a2 != -4096 && a2 != -8192 )
    {
      sub_BD73F0(a1 + 8);
      v17 = a10;
      v16 = a9;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 24) = 0;
  }
  *(_BYTE *)(a1 + 65) = v17;
  *(_QWORD *)(a1 + 80) = a1 + 104;
  *(_QWORD *)(a1 + 48) = a7;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 88) = 8;
  *(_DWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 100) = 1;
  v18 = *(_BYTE *)(a11 + 28) == 0;
  *(_QWORD *)(a1 + 32) = a3;
  *(_DWORD *)(a1 + 40) = v14;
  *(_DWORD *)(a1 + 44) = v15;
  *(_QWORD *)(a1 + 56) = a8;
  *(_BYTE *)(a1 + 64) = v16;
  *(_DWORD *)(a1 + 168) = a12;
  result = *(__int64 **)(a11 + 8);
  if ( v18 )
    v20 = *(unsigned int *)(a11 + 16);
  else
    v20 = *(unsigned int *)(a11 + 20);
  v21 = &result[v20];
  if ( v21 != result )
  {
    while ( 1 )
    {
      v22 = *result;
      v23 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v21 == ++result )
        return result;
    }
    if ( v21 != result )
    {
      v24 = 1;
LABEL_13:
      v25 = *(__int64 **)(a1 + 80);
      v26 = *(unsigned int *)(a1 + 92);
      v27 = &v25[v26];
      if ( v25 == v27 )
      {
LABEL_24:
        if ( (unsigned int)v26 < *(_DWORD *)(a1 + 88) )
        {
          v26 = (unsigned int)(v26 + 1);
          *(_DWORD *)(a1 + 92) = v26;
          *v27 = v22;
          v24 = *(_BYTE *)(a1 + 100);
          ++*(_QWORD *)(a1 + 72);
          goto LABEL_17;
        }
        goto LABEL_23;
      }
      while ( v22 != *v25 )
      {
        if ( v27 == ++v25 )
          goto LABEL_24;
      }
LABEL_17:
      while ( 1 )
      {
        result = v23 + 1;
        if ( v21 == v23 + 1 )
          break;
        v22 = *result;
        for ( ++v23; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v23 = result )
        {
          if ( v21 == ++result )
            return result;
          v22 = *result;
        }
        if ( v21 == v23 )
          return result;
        if ( v24 )
          goto LABEL_13;
LABEL_23:
        sub_C8CC70(a1 + 72, v22, (__int64)v27, v26, a5, a6);
        v24 = *(_BYTE *)(a1 + 100);
      }
    }
  }
  return result;
}
