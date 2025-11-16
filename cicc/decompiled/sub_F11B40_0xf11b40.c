// Function: sub_F11B40
// Address: 0xf11b40
//
__int64 **__fastcall sub_F11B40(__int64 **a1, __int64 *a2, __int64 a3)
{
  bool v6; // zf
  __int64 *v7; // r15
  int v8; // r14d
  int v9; // edi
  unsigned __int64 v10; // rax
  unsigned int i; // eax
  __int64 *v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // r14
  char v15; // r8
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rsi
  __int64 v19; // rax
  __int64 *v20; // r15
  __int64 *v21; // rax
  int v23; // [rsp+Ch] [rbp-44h] BYREF
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( (a2[1] & 1) != 0 )
  {
    v6 = *(_BYTE *)(a3 + 32) == 0;
    v7 = a2 + 2;
    v23 = 0;
    v8 = 3;
    if ( v6 )
      goto LABEL_3;
    goto LABEL_9;
  }
  v14 = *((unsigned int *)a2 + 6);
  v7 = (__int64 *)a2[2];
  if ( !(_DWORD)v14 )
  {
LABEL_18:
    v19 = 7 * v14;
    goto LABEL_19;
  }
  v8 = v14 - 1;
  v6 = *(_BYTE *)(a3 + 32) == 0;
  v23 = 0;
  if ( !v6 )
LABEL_9:
    v23 = *(unsigned __int16 *)(a3 + 24) | (*(_DWORD *)(a3 + 16) << 16);
LABEL_3:
  v25[0] = *(_QWORD *)(a3 + 40);
  v24 = *(_QWORD *)(a3 + 8);
  v9 = 1;
  v10 = 0xBF58476D1CE4E5B9LL
      * ((unsigned int)sub_F11290(&v24, &v23, v25)
       | ((unsigned __int64)(((unsigned int)*(_QWORD *)a3 >> 9) ^ ((unsigned int)*(_QWORD *)a3 >> 4)) << 32));
  for ( i = v8 & ((v10 >> 31) ^ v10); ; i = v8 & v13 )
  {
    v12 = &v7[7 * i];
    if ( *v12 == *(_QWORD *)a3 && *(_QWORD *)(a3 + 8) == v12[1] )
    {
      v15 = *(_BYTE *)(a3 + 32);
      if ( v15 == *((_BYTE *)v12 + 32)
        && (!v15 || *(_QWORD *)(a3 + 16) == v12[2] && *(_QWORD *)(a3 + 24) == v12[3])
        && *(_QWORD *)(a3 + 40) == v12[5] )
      {
        if ( (a2[1] & 1) != 0 )
        {
          v16 = a2 + 2;
          v17 = 28;
        }
        else
        {
          v16 = (__int64 *)a2[2];
          v17 = 7LL * *((unsigned int *)a2 + 6);
        }
        v18 = (__int64 *)*a2;
        *a1 = a2;
        a1[2] = v12;
        a1[1] = v18;
        a1[3] = &v16[v17];
        return a1;
      }
    }
    if ( *v12 == -4096 && !v12[1] && !*((_BYTE *)v12 + 32) && !v12[5] )
      break;
    v13 = v9 + i;
    ++v9;
  }
  if ( (a2[1] & 1) == 0 )
  {
    v7 = (__int64 *)a2[2];
    v14 = *((unsigned int *)a2 + 6);
    goto LABEL_18;
  }
  v7 = a2 + 2;
  v19 = 28;
LABEL_19:
  v20 = &v7[v19];
  v21 = (__int64 *)*a2;
  *a1 = a2;
  a1[2] = v20;
  a1[1] = v21;
  a1[3] = v20;
  return a1;
}
