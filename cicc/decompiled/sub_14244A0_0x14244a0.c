// Function: sub_14244A0
// Address: 0x14244a0
//
__int64 __fastcall sub_14244A0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v5; // r14d
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r10
  __int64 v9; // r11
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // r12
  unsigned __int64 v14; // r12
  int v15; // edx
  __int64 *v16; // r13
  unsigned __int64 v17; // r12
  unsigned int i; // eax
  __int64 *v19; // r12
  __int64 v20; // r15
  unsigned int v21; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = 1;
  v6 = a2[1];
  v7 = a2[3];
  v8 = a2[4];
  v9 = a2[5];
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *a2;
  v12 = a2[2];
  v13 = ((unsigned int)v9 >> 9)
      ^ ((unsigned int)v9 >> 4)
      ^ ((unsigned int)v8 >> 9)
      ^ ((unsigned int)v8 >> 4)
      ^ ((unsigned int)v7 >> 9)
      ^ ((unsigned int)v7 >> 4)
      ^ (37 * (_DWORD)v12)
      ^ ((unsigned int)v6 >> 9)
      ^ ((unsigned int)v6 >> 4);
  v14 = (((v13 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32)) - 1 - (v13 << 32)) >> 22)
      ^ ((v13 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32)) - 1 - (v13 << 32));
  v15 = v4 - 1;
  v16 = 0;
  v17 = ((9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13)))) >> 15)
      ^ (9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13))));
  for ( i = v15 & (((v17 - 1 - (v17 << 27)) >> 31) ^ (v17 - 1 - ((_DWORD)v17 << 27))); ; i = v15 & v21 )
  {
    v19 = (__int64 *)(v10 + 48LL * i);
    v20 = *v19;
    if ( *v19 == v11 && v6 == v19[1] && v12 == v19[2] && v7 == v19[3] && v8 == v19[4] && v9 == v19[5] )
    {
      *a3 = v19;
      return 1;
    }
    if ( v20 == -8 )
      break;
    if ( v20 == -16 && v19[1] == -16 && !v19[2] && !v19[3] && !v19[4] && !(v19[5] | (unsigned __int64)v16) )
      v16 = (__int64 *)(v10 + 48LL * i);
LABEL_7:
    v21 = v5 + i;
    ++v5;
  }
  if ( v19[1] != -8 || v19[2] || v19[3] || v19[4] || v19[5] )
    goto LABEL_7;
  if ( !v16 )
    v16 = (__int64 *)(v10 + 48LL * i);
  *a3 = v16;
  return 0;
}
