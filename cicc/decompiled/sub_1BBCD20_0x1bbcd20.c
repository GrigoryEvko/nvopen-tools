// Function: sub_1BBCD20
// Address: 0x1bbcd20
//
__int64 __fastcall sub_1BBCD20(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 v3; // rcx
  int v4; // r10d
  unsigned int v5; // r9d
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // eax
  int v13; // r11d

  v2 = *(_BYTE *)(a1 + 32) & 1;
  if ( v2 )
  {
    v3 = a1 + 40;
    v4 = 3;
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 48);
    v3 = *(_QWORD *)(a1 + 40);
    if ( !(_DWORD)v10 )
      goto LABEL_11;
    v4 = v10 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 == *v6 )
    goto LABEL_4;
  v12 = 1;
  while ( v7 != -8 )
  {
    v13 = v12 + 1;
    v5 = v4 & (v12 + v5);
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      goto LABEL_4;
    v12 = v13;
  }
  if ( v2 )
  {
    v11 = 64;
    goto LABEL_12;
  }
  v10 = *(unsigned int *)(a1 + 48);
LABEL_11:
  v11 = 16 * v10;
LABEL_12:
  v6 = (__int64 *)(v3 + v11);
LABEL_4:
  v8 = 64;
  if ( !v2 )
    v8 = 16LL * *(unsigned int *)(a1 + 48);
  if ( v6 == (__int64 *)(v8 + v3) )
    return 0;
  else
    return *(_QWORD *)a1 + 176LL * *((int *)v6 + 2);
}
