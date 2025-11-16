// Function: sub_B22C60
// Address: 0xb22c60
//
__int64 __fastcall sub_B22C60(__int64 a1, __int64 *a2)
{
  bool v4; // zf
  __int64 v5; // rax
  unsigned int v7; // edx
  int v8; // edx
  unsigned int v9; // esi
  unsigned int v10; // edi
  __int64 v11; // rdx
  __int64 v12; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = (unsigned __int8)sub_B1BE60(a1, a2, &v12) == 0;
  v5 = v12;
  if ( !v4 )
    return v12 + 8;
  v7 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v13[0] = v5;
  v8 = (v7 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v10 = 12;
    v9 = 4;
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 24);
    v10 = 3 * v9;
  }
  if ( 4 * v8 >= v10 )
  {
    v9 *= 2;
    goto LABEL_12;
  }
  if ( v9 - (v8 + *(_DWORD *)(a1 + 12)) <= v9 >> 3 )
  {
LABEL_12:
    sub_B228B0(a1, v9);
    sub_B1BE60(a1, a2, v13);
    v5 = v13[0];
    v8 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v8);
  if ( *(_QWORD *)v5 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v11 = *a2;
  *(_QWORD *)(v5 + 40) = v5 + 56;
  *(_QWORD *)v5 = v11;
  *(_QWORD *)(v5 + 8) = v5 + 24;
  *(_QWORD *)(v5 + 16) = 0x200000000LL;
  *(_QWORD *)(v5 + 48) = 0x200000000LL;
  *(_OWORD *)(v5 + 24) = 0;
  *(_OWORD *)(v5 + 56) = 0;
  return v5 + 8;
}
