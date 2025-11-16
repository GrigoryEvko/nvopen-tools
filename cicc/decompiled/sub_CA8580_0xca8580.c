// Function: sub_CA8580
// Address: 0xca8580
//
__int64 __fastcall sub_CA8580(__int64 a1, unsigned int *a2, unsigned int a3, _DWORD *a4, _BYTE *a5)
{
  unsigned __int64 v5; // r15
  unsigned int v6; // r12d
  _BYTE *v8; // r8
  _BYTE *v9; // rax
  __int64 v10; // rcx
  char *v11; // rsi
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 result; // rax
  __int64 v15; // rbx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  const char *v21; // [rsp+20h] [rbp-60h] BYREF
  char v22; // [rsp+40h] [rbp-40h]
  char v23; // [rsp+41h] [rbp-3Fh]

  v6 = 0;
  while ( 1 )
  {
    sub_CA7D20(a1, (char *)sub_CA60E0, 0);
    v11 = sub_CA6050(a1, *(char **)(a1 + 40));
    if ( v11 != *(char **)(a1 + 40) )
      break;
    v8 = sub_CA7C80(a1, v11);
    v9 = *(_BYTE **)(a1 + 40);
    if ( v9 != v8 && *(_DWORD *)(a1 + 60) > v6 )
    {
      v5 = *(_QWORD *)(a1 + 40);
      v6 = *(_DWORD *)(a1 + 60);
    }
    if ( v9 == *(_BYTE **)(a1 + 48) || !(unsigned __int8)sub_CA80A0(a1) )
      goto LABEL_18;
    ++*a4;
  }
  v13 = *(_DWORD *)(a1 + 60);
  if ( v13 <= a3 )
  {
LABEL_18:
    *a5 = 1;
    return 1;
  }
  *a2 = v13;
  result = 1;
  if ( v13 < v6 )
  {
    v15 = *(_QWORD *)(a1 + 336);
    v23 = 1;
    v21 = "Leading all-spaces line must be smaller than the block indent";
    v16 = *(_QWORD *)(a1 + 48);
    v22 = 3;
    if ( v5 >= v16 )
      v5 = v16 - 1;
    if ( v15 )
    {
      v17 = sub_2241E50(a1, v11, v16 - 1, v10, v12);
      *(_DWORD *)v15 = 22;
      *(_QWORD *)(v15 + 8) = v17;
    }
    if ( !*(_BYTE *)(a1 + 75) )
      sub_C91CB0(*(__int64 **)a1, v5, 0, (__int64)&v21, 0, 0, 0, 0, *(_BYTE *)(a1 + 76));
    *(_BYTE *)(a1 + 75) = 1;
    return 0;
  }
  return result;
}
