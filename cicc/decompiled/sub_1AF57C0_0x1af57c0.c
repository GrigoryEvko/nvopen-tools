// Function: sub_1AF57C0
// Address: 0x1af57c0
//
__int64 __fastcall sub_1AF57C0(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v4; // r8
  int v5; // edi
  __int64 v6; // rcx
  __int64 v7; // rsi
  __int64 v8; // r9
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // r10
  int v11; // ebx
  unsigned __int64 v12; // rax
  __int64 *v13; // r12
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r10
  unsigned int i; // eax
  __int64 *v17; // r10
  __int64 v18; // r11
  unsigned int v19; // eax
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = a1 + 16;
    v5 = 7;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = result - 1;
    if ( !(_DWORD)result )
    {
      *a3 = 0;
      return result;
    }
  }
  v6 = *a2;
  v7 = a2[1];
  v8 = a2[2];
  v9 = (((((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
        | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32)) >> 22)
     ^ ((((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
       | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))
      - 1
      - ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32));
  v10 = ((9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13)))) >> 15)
      ^ (9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13))));
  v11 = 1;
  v12 = (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
       | ((((v10 - 1 - (v10 << 27)) >> 31) ^ (v10 - 1 - (v10 << 27))) << 32))
      - 1
      - ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32);
  v13 = 0;
  v14 = ((v12 >> 22) ^ v12) - 1 - (((v12 >> 22) ^ v12) << 13);
  v15 = ((9 * ((v14 >> 8) ^ v14)) >> 15) ^ (9 * ((v14 >> 8) ^ v14));
  for ( i = v5 & (((v15 - 1 - (v15 << 27)) >> 31) ^ (v15 - 1 - ((_DWORD)v15 << 27))); ; i = v5 & v19 )
  {
    v17 = (__int64 *)(v4 + 24LL * i);
    v18 = *v17;
    if ( *v17 == v6 && v7 == v17[1] && v8 == v17[2] )
    {
      *a3 = v17;
      return 1;
    }
    if ( v18 == -8 )
      break;
    if ( v18 == -16 && v17[1] == -16 && v17[2] == -16 && !v13 )
      v13 = (__int64 *)(v4 + 24LL * i);
LABEL_8:
    v19 = v11 + i;
    ++v11;
  }
  if ( v17[1] != -8 || v17[2] != -8 )
    goto LABEL_8;
  if ( !v13 )
    v13 = (__int64 *)(v4 + 24LL * i);
  result = 0;
  *a3 = v13;
  return result;
}
