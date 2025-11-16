// Function: sub_1B74870
// Address: 0x1b74870
//
__int64 __fastcall sub_1B74870(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  char v3; // dl
  __int64 v4; // rax
  int v5; // r9d
  unsigned int v6; // edi
  __int64 *v7; // rcx
  __int64 v8; // r10
  __int64 v9; // rsi
  unsigned int v10; // r8d
  __int64 v12; // rcx
  __int64 v13; // rcx
  int v14; // ecx
  int v15; // r11d

  v2 = *a1;
  v3 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v3 )
  {
    v4 = v2 + 16;
    v5 = 31;
  }
  else
  {
    v12 = *(unsigned int *)(v2 + 24);
    v4 = *(_QWORD *)(v2 + 16);
    if ( !(_DWORD)v12 )
      goto LABEL_12;
    v5 = v12 - 1;
  }
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 24LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
    goto LABEL_4;
  v14 = 1;
  while ( v8 != -4 )
  {
    v15 = v14 + 1;
    v6 = v5 & (v14 + v6);
    v7 = (__int64 *)(v4 + 24LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      goto LABEL_4;
    v14 = v15;
  }
  if ( v3 )
  {
    v13 = 768;
    goto LABEL_13;
  }
  v12 = *(unsigned int *)(v2 + 24);
LABEL_12:
  v13 = 24 * v12;
LABEL_13:
  v7 = (__int64 *)(v4 + v13);
LABEL_4:
  v9 = 768;
  if ( !v3 )
    v9 = 24LL * *(unsigned int *)(v2 + 24);
  v10 = 0;
  if ( v7 != (__int64 *)(v9 + v4) )
    return *((unsigned __int8 *)v7 + 8);
  return v10;
}
