// Function: sub_BCF150
// Address: 0xbcf150
//
__int64 __fastcall sub_BCF150(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  _QWORD *v4; // rdx
  __int64 result; // rax
  char v6; // dl
  __int64 *v7; // rbx
  __int64 *v8; // r12
  unsigned int v9; // edx
  unsigned int v10; // ecx

  if ( !*(_BYTE *)(a2 + 28) )
    goto LABEL_7;
  v2 = *(_QWORD **)(a2 + 8);
  v3 = *(unsigned int *)(a2 + 20);
  v4 = &v2[v3];
  if ( v2 != v4 )
  {
    while ( a1 != *v2 )
    {
      if ( v4 == ++v2 )
        goto LABEL_13;
    }
    return 0;
  }
LABEL_13:
  if ( (unsigned int)v3 < *(_DWORD *)(a2 + 16) )
  {
    *(_DWORD *)(a2 + 20) = v3 + 1;
    *v4 = a1;
    ++*(_QWORD *)a2;
  }
  else
  {
LABEL_7:
    sub_C8CC70(a2, a1);
    if ( !v6 )
      return 0;
  }
  v7 = *(__int64 **)(a1 + 16);
  v8 = &v7[*(unsigned int *)(a1 + 12)];
  if ( v7 == v8 )
  {
LABEL_15:
    v9 = *(_DWORD *)(a1 + 8);
    v10 = v9 >> 8;
    result = (v9 & 0x100) == 0;
    if ( (v9 & 0x100) != 0 )
    {
      LOBYTE(v10) = BYTE1(v9) | 0x80;
      *(_DWORD *)(a1 + 8) = (v10 << 8) | (unsigned __int8)v9;
      return result;
    }
    return 0;
  }
  while ( 1 )
  {
    result = sub_BCF080(*v7);
    if ( (_BYTE)result )
      break;
    if ( v8 == ++v7 )
      goto LABEL_15;
  }
  *(_DWORD *)(a1 + 8) |= 0x4000u;
  return result;
}
