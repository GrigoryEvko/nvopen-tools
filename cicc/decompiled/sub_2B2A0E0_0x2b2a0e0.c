// Function: sub_2B2A0E0
// Address: 0x2b2a0e0
//
__int64 __fastcall sub_2B2A0E0(__int64 a1, __int64 a2)
{
  char v2; // cl
  __int64 v3; // r9
  int v4; // eax
  unsigned int v5; // edx
  __int64 *v6; // r8
  __int64 v7; // r10
  __int64 v8; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  int v12; // r8d
  int v13; // r11d

  v2 = *(_BYTE *)(a1 + 88) & 1;
  if ( v2 )
  {
    v3 = a1 + 96;
    v4 = 3;
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 104);
    v3 = *(_QWORD *)(a1 + 96);
    if ( !(_DWORD)v10 )
      goto LABEL_12;
    v4 = v10 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 72LL * v5);
  v7 = *v6;
  if ( a2 == *v6 )
    goto LABEL_4;
  v12 = 1;
  while ( v7 != -4096 )
  {
    v13 = v12 + 1;
    v5 = v4 & (v12 + v5);
    v6 = (__int64 *)(v3 + 72LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      goto LABEL_4;
    v12 = v13;
  }
  if ( v2 )
  {
    v11 = 288;
    goto LABEL_13;
  }
  v10 = *(unsigned int *)(a1 + 104);
LABEL_12:
  v11 = 72 * v10;
LABEL_13:
  v6 = (__int64 *)(v3 + v11);
LABEL_4:
  v8 = 288;
  if ( !v2 )
    v8 = 72LL * *(unsigned int *)(a1 + 104);
  if ( v6 == (__int64 *)(v3 + v8) )
    return 0;
  else
    return v6[1];
}
