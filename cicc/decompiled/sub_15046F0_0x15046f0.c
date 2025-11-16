// Function: sub_15046F0
// Address: 0x15046f0
//
__int64 *__fastcall sub_15046F0(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // r12
  int v6; // eax
  int v7; // edx
  __int64 v8; // rsi
  unsigned int v9; // eax
  __int64 v10; // rcx
  int v12; // edi
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16[2]; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+10h] [rbp-30h]
  char v18; // [rsp+11h] [rbp-2Fh]

  if ( *(_BYTE *)(a2 + 1657) )
  {
LABEL_16:
    *a1 = 1;
    return a1;
  }
  *(_BYTE *)(a2 + 1657) = 1;
  while ( 2 )
  {
    v4 = *(__int64 **)(a2 + 1592);
    if ( v4 == *(__int64 **)(a2 + 1624) )
    {
LABEL_15:
      *(_BYTE *)(a2 + 1657) = 0;
      goto LABEL_16;
    }
    while ( 1 )
    {
      v5 = *v4;
      if ( v4 == (__int64 *)(*(_QWORD *)(a2 + 1608) - 8LL) )
      {
        j_j___libc_free_0(*(_QWORD *)(a2 + 1600), 512);
        v13 = (__int64 *)(*(_QWORD *)(a2 + 1616) + 8LL);
        *(_QWORD *)(a2 + 1616) = v13;
        v14 = *v13;
        v15 = *v13 + 512;
        *(_QWORD *)(a2 + 1600) = v14;
        *(_QWORD *)(a2 + 1608) = v15;
        *(_QWORD *)(a2 + 1592) = v14;
      }
      else
      {
        *(_QWORD *)(a2 + 1592) = v4 + 1;
      }
      v6 = *(_DWORD *)(a2 + 1568);
      if ( v6 )
        break;
LABEL_14:
      v4 = *(__int64 **)(a2 + 1592);
      if ( *(__int64 **)(a2 + 1624) == v4 )
        goto LABEL_15;
    }
    v7 = v6 - 1;
    v8 = *(_QWORD *)(a2 + 1552);
    v9 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = *(_QWORD *)(v8 + 32LL * v9);
    if ( v5 != v10 )
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v9 = v7 & (v12 + v9);
        v10 = *(_QWORD *)(v8 + 32LL * v9);
        if ( v5 == v10 )
          goto LABEL_8;
        ++v12;
      }
      goto LABEL_14;
    }
LABEL_8:
    if ( (*(_BYTE *)(v5 + 34) & 0x40) != 0 )
    {
      sub_1503DC0(v16, a2, v5);
      if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = v16[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        return a1;
      }
      continue;
    }
    break;
  }
  v18 = 1;
  v17 = 3;
  v16[0] = (unsigned __int64)"Never resolved function from blockaddress";
  sub_14EE4B0(a1, a2 + 8, (__int64)v16);
  return a1;
}
