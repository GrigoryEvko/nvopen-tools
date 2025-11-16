// Function: sub_15D0EA0
// Address: 0x15d0ea0
//
__int64 *__fastcall sub_15D0EA0(__int64 a1, __int64 *a2)
{
  char v3; // al
  __int64 *v4; // r8
  unsigned int v6; // eax
  int v7; // eax
  unsigned int v8; // esi
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 *v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_15D0A10(a1, a2, v11);
  v4 = v11[0];
  if ( v3 )
    return v11[0];
  v6 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
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
  if ( v9 <= 4 * v7 )
  {
    v8 *= 2;
  }
  else if ( v8 - (v7 + *(_DWORD *)(a1 + 12)) > v8 >> 3 )
  {
    goto LABEL_7;
  }
  sub_15D0B40(a1, v8);
  sub_15D0A10(a1, a2, v11);
  v4 = v11[0];
  v7 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
LABEL_7:
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v7);
  if ( *v4 != -8 || v4[1] != -8 )
    --*(_DWORD *)(a1 + 12);
  *v4 = *a2;
  v10 = a2[1];
  *((_DWORD *)v4 + 4) = 0;
  v4[1] = v10;
  return v4;
}
