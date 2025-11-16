// Function: sub_2AC7E90
// Address: 0x2ac7e90
//
_OWORD *__fastcall sub_2AC7E90(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rax
  _OWORD *result; // rax
  int v6; // ecx
  unsigned int v7; // esi
  int v8; // edx
  char v9; // dl
  __int64 v10; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_2ABE520(a1, (__int64 *)a2, &v10) == 0;
  v4 = v10;
  if ( !v3 )
    return (_OWORD *)(v10 + 16);
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v11[0] = v10;
  ++*(_QWORD *)a1;
  v8 = v6 + 1;
  if ( 4 * (v6 + 1) >= 3 * v7 )
  {
    v7 *= 2;
  }
  else if ( v7 - *(_DWORD *)(a1 + 20) - v8 > v7 >> 3 )
  {
    goto LABEL_5;
  }
  sub_2AC7CC0(a1, v7);
  sub_2ABE520(a1, (__int64 *)a2, v11);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  v4 = v11[0];
LABEL_5:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *(_QWORD *)v4 != -4096 || *(_DWORD *)(v4 + 8) != -1 || !*(_BYTE *)(v4 + 12) )
    --*(_DWORD *)(a1 + 20);
  result = (_OWORD *)(v4 + 16);
  *((_QWORD *)result - 2) = *(_QWORD *)a2;
  *((_DWORD *)result - 2) = *(_DWORD *)(a2 + 8);
  v9 = *(_BYTE *)(a2 + 12);
  *result = 0;
  *((_BYTE *)result - 4) = v9;
  result[1] = 0;
  result[2] = 0;
  return result;
}
