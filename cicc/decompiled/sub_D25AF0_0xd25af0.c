// Function: sub_D25AF0
// Address: 0xd25af0
//
__int64 __fastcall sub_D25AF0(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  __int64 v4; // rax
  __int64 result; // rax
  unsigned int v6; // edx
  int v7; // edx
  unsigned int v8; // esi
  unsigned int v9; // ecx
  __int64 v10; // rdx
  __int64 v11; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v12[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_D24B80(a1, a2, &v11) == 0;
  v4 = v11;
  if ( !v3 )
    return v11 + 8;
  v6 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v12[0] = v4;
  v7 = (v6 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v9 = 12;
    v8 = 4;
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 24);
    v9 = 3 * v8;
  }
  if ( 4 * v7 >= v9 )
  {
    v8 *= 2;
    goto LABEL_12;
  }
  if ( v8 - (v7 + *(_DWORD *)(a1 + 12)) <= v8 >> 3 )
  {
LABEL_12:
    sub_D257F0(a1, v8);
    sub_D24B80(a1, a2, v12);
    v4 = v12[0];
    v7 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v7);
  if ( *(_QWORD *)v4 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v10 = *a2;
  *(_DWORD *)(v4 + 8) = 0;
  result = v4 + 8;
  *(_QWORD *)(result - 8) = v10;
  return result;
}
