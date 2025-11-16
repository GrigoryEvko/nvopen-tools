// Function: sub_15D45B0
// Address: 0x15d45b0
//
char *__fastcall sub_15D45B0(__int64 a1)
{
  __int64 v1; // r14
  char v2; // r8
  char *result; // rax
  _BYTE *v4; // rsi
  int v5; // ecx
  unsigned int v6; // esi
  int v7; // edx
  __int64 v8; // rdx
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = a1 + 24;
  v9 = 0;
  v2 = sub_15CEFE0(a1 + 24, &v9, v10);
  result = (char *)v10[0];
  if ( v2 )
    goto LABEL_2;
  v5 = *(_DWORD *)(a1 + 40);
  v6 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 24);
  v7 = v5 + 1;
  if ( 4 * (v5 + 1) >= 3 * v6 )
  {
    v6 *= 2;
    goto LABEL_13;
  }
  if ( v6 - *(_DWORD *)(a1 + 44) - v7 <= v6 >> 3 )
  {
LABEL_13:
    sub_15D4370(v1, v6);
    sub_15CEFE0(v1, &v9, v10);
    result = (char *)v10[0];
    v7 = *(_DWORD *)(a1 + 40) + 1;
  }
  *(_DWORD *)(a1 + 40) = v7;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 44);
  v8 = v9;
  *((_QWORD *)result + 6) = 0x200000000LL;
  *(_QWORD *)result = v8;
  *((_QWORD *)result + 5) = result + 56;
  *(_OWORD *)(result + 8) = 0;
  *(_OWORD *)(result + 24) = 0;
  *(_OWORD *)(result + 56) = 0;
LABEL_2:
  *((_DWORD *)result + 4) = 1;
  *((_DWORD *)result + 2) = 1;
  *((_QWORD *)result + 3) = 0;
  v4 = *(_BYTE **)(a1 + 8);
  v10[0] = 0;
  if ( v4 == *(_BYTE **)(a1 + 16) )
    return sub_15D0700(a1, v4, v10);
  if ( v4 )
  {
    *(_QWORD *)v4 = 0;
    v4 = *(_BYTE **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v4 + 8;
  return result;
}
