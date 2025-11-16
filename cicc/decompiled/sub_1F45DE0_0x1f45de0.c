// Function: sub_1F45DE0
// Address: 0x1f45de0
//
__int64 __fastcall sub_1F45DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v8; // rbx
  unsigned int v9; // edx
  __int64 result; // rax
  char v11; // [rsp+8h] [rbp-28h]
  char v12; // [rsp+Ch] [rbp-24h]

  v8 = *(_QWORD *)(a1 + 216);
  v9 = *(_DWORD *)(v8 + 40);
  if ( v9 >= *(_DWORD *)(v8 + 44) )
  {
    v11 = a6;
    v12 = a5;
    sub_16CD150(v8 + 32, (const void *)(v8 + 48), 0, 32, a5, a6);
    v9 = *(_DWORD *)(v8 + 40);
    a6 = v11;
    a5 = v12;
  }
  result = *(_QWORD *)(v8 + 32) + 32LL * v9;
  if ( result )
  {
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = a3;
    *(_QWORD *)(result + 16) = a4;
    *(_BYTE *)(result + 24) = a5;
    *(_BYTE *)(result + 25) = a6;
    v9 = *(_DWORD *)(v8 + 40);
  }
  *(_DWORD *)(v8 + 40) = v9 + 1;
  return result;
}
