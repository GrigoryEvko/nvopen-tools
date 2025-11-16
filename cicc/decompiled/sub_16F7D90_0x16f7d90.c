// Function: sub_16F7D90
// Address: 0x16f7d90
//
__int64 __fastcall sub_16F7D90(__int64 a1, _DWORD *a2, unsigned int a3, _DWORD *a4, _BYTE *a5)
{
  unsigned int v7; // ebx
  _BYTE *v8; // r8
  _BYTE *v9; // rax
  __int64 v10; // rcx
  char *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 result; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  const char *v20; // [rsp+10h] [rbp-50h] BYREF
  char v21; // [rsp+20h] [rbp-40h]
  char v22; // [rsp+21h] [rbp-3Fh]

  v7 = 0;
  while ( 1 )
  {
    sub_16F77C0(a1, (char *)sub_16F6410, 0);
    v11 = sub_16F6380(a1, *(char **)(a1 + 40));
    if ( v11 != *(char **)(a1 + 40) )
      break;
    v8 = sub_16F7720(a1, v11);
    v9 = *(_BYTE **)(a1 + 40);
    if ( v9 != v8 && v7 < *(_DWORD *)(a1 + 60) )
      v7 = *(_DWORD *)(a1 + 60);
    if ( v9 == *(_BYTE **)(a1 + 48) || !(unsigned __int8)sub_16F7970(a1) )
      goto LABEL_18;
    ++*a4;
  }
  v13 = *(unsigned int *)(a1 + 60);
  if ( (unsigned int)v13 <= a3 )
  {
LABEL_18:
    *a5 = 1;
    return 1;
  }
  *a2 = v13;
  result = 1;
  if ( (unsigned int)v13 < v7 )
  {
    v22 = 1;
    v20 = "Leading all-spaces line must be smaller than the block indent";
    v15 = *(_QWORD *)(a1 + 48);
    v21 = 3;
    if ( *(_QWORD *)(a1 + 40) >= v15 )
      *(_QWORD *)(a1 + 40) = v15 - 1;
    v16 = *(_QWORD *)(a1 + 344);
    if ( v16 )
    {
      v17 = sub_2241E50(a1, v11, v13, v10, v12);
      *(_DWORD *)v16 = 22;
      *(_QWORD *)(v16 + 8) = v17;
    }
    if ( !*(_BYTE *)(a1 + 74) )
      sub_16D14E0(*(__int64 **)a1, *(_QWORD *)(a1 + 40), 0, (__int64)&v20, 0, 0, 0, 0, *(_BYTE *)(a1 + 75));
    *(_BYTE *)(a1 + 74) = 1;
    return 0;
  }
  return result;
}
