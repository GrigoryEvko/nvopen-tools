// Function: sub_9FF010
// Address: 0x9ff010
//
__int64 *__fastcall sub_9FF010(__int64 *a1, __int64 a2)
{
  _BYTE **i; // rax
  _BYTE *v5; // r13
  int v6; // eax
  __int64 v7; // rsi
  int v8; // ecx
  unsigned int v9; // eax
  _BYTE *v10; // rdx
  unsigned __int64 v11; // rax
  int v13; // edi
  _BYTE **v14; // r14
  _BYTE **v15; // r13
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20[4]; // [rsp+10h] [rbp-50h] BYREF
  char v21; // [rsp+30h] [rbp-30h]
  char v22; // [rsp+31h] [rbp-2Fh]

  if ( *(_BYTE *)(a2 + 1833) )
    goto LABEL_21;
  *(_BYTE *)(a2 + 1833) = 1;
LABEL_3:
  for ( i = *(_BYTE ***)(a2 + 1744); i != *(_BYTE ***)(a2 + 1776); i = *(_BYTE ***)(a2 + 1744) )
  {
    v5 = *i;
    if ( i == (_BYTE **)(*(_QWORD *)(a2 + 1760) - 8LL) )
    {
      j_j___libc_free_0(*(_QWORD *)(a2 + 1752), 512);
      v17 = (__int64 *)(*(_QWORD *)(a2 + 1768) + 8LL);
      *(_QWORD *)(a2 + 1768) = v17;
      v18 = *v17;
      v19 = *v17 + 512;
      *(_QWORD *)(a2 + 1752) = v18;
      *(_QWORD *)(a2 + 1760) = v19;
      *(_QWORD *)(a2 + 1744) = v18;
    }
    else
    {
      *(_QWORD *)(a2 + 1744) = i + 1;
    }
    v6 = *(_DWORD *)(a2 + 1720);
    v7 = *(_QWORD *)(a2 + 1704);
    if ( v6 )
    {
      v8 = v6 - 1;
      v9 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v10 = *(_BYTE **)(v7 + 32LL * v9);
      if ( v5 == v10 )
      {
LABEL_8:
        if ( (v5[35] & 8) == 0 )
        {
          v22 = 1;
          v21 = 3;
          v20[0] = (__int64)"Never resolved function from blockaddress";
          sub_9C81F0(a1, a2 + 8, (__int64)v20);
          return a1;
        }
        sub_9FDC80(v20, a2, v5);
        v11 = v20[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v20[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_10;
        goto LABEL_3;
      }
      v13 = 1;
      while ( v10 != (_BYTE *)-4096LL )
      {
        v9 = v8 & (v13 + v9);
        v10 = *(_BYTE **)(v7 + 32LL * v9);
        if ( v5 == v10 )
          goto LABEL_8;
        ++v13;
      }
    }
  }
  v14 = *(_BYTE ***)(a2 + 1808);
  v15 = *(_BYTE ***)(a2 + 1816);
  if ( v15 != v14 )
  {
    while ( 1 )
    {
      sub_9FDC80(v20, a2, *v14);
      v11 = v20[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v20[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( v15 == ++v14 )
      {
        v16 = *(_QWORD *)(a2 + 1808);
        if ( v16 != *(_QWORD *)(a2 + 1816) )
          *(_QWORD *)(a2 + 1816) = v16;
        goto LABEL_20;
      }
    }
LABEL_10:
    *a1 = v11 | 1;
    return a1;
  }
LABEL_20:
  *(_BYTE *)(a2 + 1833) = 0;
LABEL_21:
  *a1 = 1;
  return a1;
}
