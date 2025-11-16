// Function: sub_38C7080
// Address: 0x38c7080
//
__int64 __fastcall sub_38C7080(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // r14
  int v7; // r13d
  int v8; // eax
  __int64 v9; // r9
  int v10; // r8d
  unsigned int i; // edx
  _QWORD *v12; // rax
  unsigned int v13; // edx
  __int64 v14; // r11
  __int64 v15; // [rsp+10h] [rbp-60h]
  __int64 v16; // [rsp+30h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v4 - 1;
  v8 = sub_38C6F70(
         (__int64 *)a2,
         (int *)(a2 + 8),
         (int *)(a2 + 12),
         (char *)(a2 + 16),
         (char *)(a2 + 17),
         (int *)(a2 + 20));
  v9 = 0;
  v10 = 1;
  for ( i = v7 & v8; ; i = v7 & v13 )
  {
    v12 = (_QWORD *)(v6 + 32LL * i);
    if ( *(_QWORD *)a2 == *v12
      && *(_QWORD *)(a2 + 8) == v12[1]
      && ((v12[2] ^ *(_QWORD *)(a2 + 16)) & 0xFFFFFFFF0000FFFFLL) == 0 )
    {
      *a3 = v12;
      return 1;
    }
    if ( !*v12 )
      break;
LABEL_6:
    v13 = v10 + i;
    ++v10;
  }
  v14 = v12[1];
  if ( v14 != 0xFFFFFFFF00000000LL )
  {
    if ( v14 == 0xFFFFFFFFLL )
    {
      LOWORD(v16) = 0;
      HIDWORD(v16) = 0x7FFFFFFF;
      if ( !(v9 | (v16 ^ v12[2]) & 0xFFFFFFFF0000FFFFLL) )
        v9 = v6 + 32LL * i;
    }
    goto LABEL_6;
  }
  HIDWORD(v15) = 0x7FFFFFFF;
  LOWORD(v15) = 0;
  if ( ((v15 ^ v12[2]) & 0xFFFFFFFF0000FFFFLL) != 0 )
    goto LABEL_6;
  if ( !v9 )
    v9 = v6 + 32LL * i;
  *a3 = v9;
  return 0;
}
