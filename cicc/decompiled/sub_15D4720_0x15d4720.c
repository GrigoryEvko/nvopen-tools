// Function: sub_15D4720
// Address: 0x15d4720
//
__int64 __fastcall sub_15D4720(__int64 a1, __int64 *a2)
{
  char v3; // r8
  __int64 result; // rax
  int v5; // ecx
  unsigned int v6; // esi
  int v7; // edx
  __int64 v8; // rdx
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_15CEFE0(a1, a2, v9);
  result = v9[0];
  if ( !v3 )
  {
    v5 = *(_DWORD *)(a1 + 16);
    v6 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)a1;
    v7 = v5 + 1;
    if ( 4 * (v5 + 1) >= 3 * v6 )
    {
      sub_15D4370(a1, 2 * v6);
      sub_15CEFE0(a1, a2, v9);
      result = v9[0];
      v7 = *(_DWORD *)(a1 + 16) + 1;
    }
    else if ( v6 - *(_DWORD *)(a1 + 20) - v7 <= v6 >> 3 )
    {
      sub_15D4370(a1, v6);
      sub_15CEFE0(a1, a2, v9);
      result = v9[0];
      v7 = *(_DWORD *)(a1 + 16) + 1;
    }
    *(_DWORD *)(a1 + 16) = v7;
    if ( *(_QWORD *)result != -8 )
      --*(_DWORD *)(a1 + 20);
    v8 = *a2;
    *(_QWORD *)(result + 48) = 0x200000000LL;
    *(_QWORD *)result = v8;
    *(_QWORD *)(result + 40) = result + 56;
    *(_OWORD *)(result + 8) = 0;
    *(_OWORD *)(result + 24) = 0;
    *(_OWORD *)(result + 56) = 0;
  }
  return result;
}
