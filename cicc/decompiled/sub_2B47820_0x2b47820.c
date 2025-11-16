// Function: sub_2B47820
// Address: 0x2b47820
//
__int64 __fastcall sub_2B47820(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v5; // r8
  int v6; // r9d
  __int64 *v7; // r11
  __int64 v8; // rsi
  int v9; // ebx
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  unsigned int i; // eax
  __int64 *v13; // rdx
  __int64 v14; // r10
  unsigned int v15; // eax
  int v16; // edx

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = a1 + 16;
    v6 = 7;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = v16 - 1;
    if ( !v16 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v7 = 0;
  v8 = *a2;
  v9 = 1;
  v10 = a2[1];
  v11 = 0xBF58476D1CE4E5B9LL
      * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
       | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32));
  for ( i = v6 & ((v11 >> 31) ^ v11); ; i = v6 & v15 )
  {
    v13 = (__int64 *)(v5 + 24LL * i);
    v14 = *v13;
    if ( *v13 == v8 && v13[1] == v10 )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && v13[1] == -8192 && !v7 )
      v7 = (__int64 *)(v5 + 24LL * i);
LABEL_10:
    v15 = v9 + i;
    ++v9;
  }
  if ( v13[1] != -4096 )
    goto LABEL_10;
  if ( !v7 )
    v7 = (__int64 *)(v5 + 24LL * i);
  *a3 = v7;
  return 0;
}
