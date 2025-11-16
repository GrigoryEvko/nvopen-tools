// Function: sub_2E726F0
// Address: 0x2e726f0
//
__int64 __fastcall sub_2E726F0(__int64 a1, __int64 *a2)
{
  bool v4; // zf
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r8
  unsigned int v9; // edx
  unsigned __int64 v10; // rdx
  unsigned int v11; // esi
  unsigned int v12; // edi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v16[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = (unsigned __int8)sub_2E6E4E0(a1, a2, &v15) == 0;
  v6 = v15;
  v7 = v15 + 8;
  if ( !v4 )
    return v15 + 8;
  v9 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v16[0] = v6;
  v10 = (v9 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v12 = 12;
    v11 = 4;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v12 = 3 * v11;
  }
  v13 = (unsigned int)(4 * v10);
  if ( (unsigned int)v13 >= v12 )
  {
    v11 *= 2;
    goto LABEL_12;
  }
  v13 = v11 >> 3;
  if ( v11 - ((_DWORD)v10 + *(_DWORD *)(a1 + 12)) <= (unsigned int)v13 )
  {
LABEL_12:
    sub_2E66810(a1, v11, v10, v13, v7, v5);
    sub_2E6E4E0(a1, a2, v16);
    v6 = v16[0];
    LODWORD(v10) = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v10);
  if ( *(_QWORD *)v6 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v14 = *a2;
  *(_QWORD *)(v6 + 40) = v6 + 56;
  *(_QWORD *)v6 = v14;
  *(_QWORD *)(v6 + 8) = v6 + 24;
  *(_QWORD *)(v6 + 16) = 0x200000000LL;
  *(_QWORD *)(v6 + 48) = 0x200000000LL;
  *(_OWORD *)(v6 + 24) = 0;
  *(_OWORD *)(v6 + 56) = 0;
  return v6 + 8;
}
