// Function: sub_1BA3780
// Address: 0x1ba3780
//
__int64 __fastcall sub_1BA3780(__int64 a1, __int64 a2)
{
  char v3; // r8
  __int64 result; // rax
  int v5; // ecx
  unsigned int v6; // esi
  int v7; // edx
  int v8; // edx
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_1B99450(a1, (__int64 *)a2, v9);
  result = v9[0];
  if ( !v3 )
  {
    v5 = *(_DWORD *)(a1 + 16);
    v6 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)a1;
    v7 = v5 + 1;
    if ( 4 * (v5 + 1) >= 3 * v6 )
    {
      sub_1BA2A10(a1, 2 * v6);
      sub_1B99450(a1, (__int64 *)a2, v9);
      result = v9[0];
      v7 = *(_DWORD *)(a1 + 16) + 1;
    }
    else if ( v6 - *(_DWORD *)(a1 + 20) - v7 <= v6 >> 3 )
    {
      sub_1BA2A10(a1, v6);
      sub_1B99450(a1, (__int64 *)a2, v9);
      result = v9[0];
      v7 = *(_DWORD *)(a1 + 16) + 1;
    }
    *(_DWORD *)(a1 + 16) = v7;
    if ( *(_QWORD *)result != -8 || *(_DWORD *)(result + 8) != -1 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)result = *(_QWORD *)a2;
    v8 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)(result + 16) = 0;
    *(_DWORD *)(result + 8) = v8;
  }
  return result;
}
