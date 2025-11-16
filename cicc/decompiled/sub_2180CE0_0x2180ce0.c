// Function: sub_2180CE0
// Address: 0x2180ce0
//
__int64 __fastcall sub_2180CE0(__int64 a1, __int64 a2, __int64 *a3)
{
  char v5; // al
  __int64 *v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  char v9; // al
  unsigned int v11; // r8d
  int v12; // eax
  int v13; // eax
  int v14; // esi
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = sub_1E1F3B0(a2, a3, v15);
  v6 = (__int64 *)v15[0];
  if ( v5 )
  {
    v7 = *(_QWORD *)a2;
    v8 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
    v9 = 0;
    goto LABEL_3;
  }
  v11 = *(_DWORD *)(a2 + 24);
  v12 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v13 = v12 + 1;
  if ( 4 * v13 >= 3 * v11 )
  {
    v14 = 2 * v11;
LABEL_10:
    sub_1E22DE0(a2, v14);
    sub_1E1F3B0(a2, a3, v15);
    v6 = (__int64 *)v15[0];
    v13 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_6;
  }
  if ( v11 - *(_DWORD *)(a2 + 20) - v13 <= v11 >> 3 )
  {
    v14 = v11;
    goto LABEL_10;
  }
LABEL_6:
  *(_DWORD *)(a2 + 16) = v13;
  if ( *v6 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v6 = *a3;
  v7 = *(_QWORD *)a2;
  v8 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
  v9 = 1;
LABEL_3:
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 32) = v9;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = v6;
  *(_QWORD *)(a1 + 24) = v8;
  return a1;
}
