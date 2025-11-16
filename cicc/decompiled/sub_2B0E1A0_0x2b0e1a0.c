// Function: sub_2B0E1A0
// Address: 0x2b0e1a0
//
__int64 __fastcall sub_2B0E1A0(__int64 a1, __int64 a2)
{
  char v2; // r9
  __int64 v3; // r8
  int v4; // r10d
  unsigned int v5; // ecx
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  int v13; // eax
  int v14; // r11d

  v2 = *(_BYTE *)(a1 + 8) & 1;
  if ( v2 )
  {
    v3 = a1 + 16;
    v4 = 15;
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v11 )
      goto LABEL_13;
    v4 = v11 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 == *v6 )
    goto LABEL_4;
  v13 = 1;
  while ( v7 != -4096 )
  {
    v14 = v13 + 1;
    v5 = v4 & (v13 + v5);
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      goto LABEL_4;
    v13 = v14;
  }
  if ( v2 )
  {
    v12 = 256;
    goto LABEL_14;
  }
  v11 = *(unsigned int *)(a1 + 24);
LABEL_13:
  v12 = 16 * v11;
LABEL_14:
  v6 = (__int64 *)(v3 + v12);
LABEL_4:
  v8 = 256;
  if ( !v2 )
    v8 = 16LL * *(unsigned int *)(a1 + 24);
  if ( v6 == (__int64 *)(v3 + v8) )
    v9 = 16LL * *(unsigned int *)(a1 + 280);
  else
    v9 = 16LL * *((unsigned int *)v6 + 2);
  return *(_QWORD *)(a1 + 272) + v9;
}
