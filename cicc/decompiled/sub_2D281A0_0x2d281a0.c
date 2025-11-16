// Function: sub_2D281A0
// Address: 0x2d281a0
//
__int64 __fastcall sub_2D281A0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rsi
  int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 j; // rdx
  unsigned int v10; // ecx
  unsigned int v11; // eax
  _QWORD *v12; // rdi
  int v13; // ebx
  __int64 v14; // rdx
  __int64 i; // rdx

  v2 = *(unsigned int *)(a1 + 64);
  v3 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 8) = 0;
  v4 = v3 + 32 * v2;
  while ( v3 != v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 - 16);
      v4 -= 32;
      if ( !v5 )
        break;
      sub_B91220(v4 + 16, v5);
      if ( v3 == v4 )
        goto LABEL_5;
    }
  }
LABEL_5:
  v6 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  *(_DWORD *)(a1 + 64) = 0;
  if ( v6 )
  {
    v10 = 4 * v6;
    v8 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v10 = 64;
    if ( (unsigned int)v8 <= v10 )
      goto LABEL_8;
    v11 = v6 - 1;
    if ( v11 )
    {
      _BitScanReverse(&v11, v11);
      v12 = *(_QWORD **)(a1 + 120);
      v13 = 1 << (33 - (v11 ^ 0x1F));
      if ( v13 < 64 )
        v13 = 64;
      if ( (_DWORD)v8 == v13 )
      {
        *(_QWORD *)(a1 + 128) = 0;
        result = (__int64)&v12[2 * (unsigned int)v8];
        do
        {
          if ( v12 )
            *v12 = -4096;
          v12 += 2;
        }
        while ( (_QWORD *)result != v12 );
        goto LABEL_11;
      }
    }
    else
    {
      v12 = *(_QWORD **)(a1 + 120);
      v13 = 64;
    }
    sub_C7D6A0((__int64)v12, 16LL * (unsigned int)v8, 8);
    result = sub_AF1560(4 * v13 / 3u + 1);
    *(_DWORD *)(a1 + 136) = result;
    if ( !(_DWORD)result )
      goto LABEL_25;
    result = sub_C7D670(16LL * (unsigned int)result, 8);
    v14 = *(unsigned int *)(a1 + 136);
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 120) = result;
    for ( i = result + 16 * v14; i != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  else
  {
    result = *(unsigned int *)(a1 + 132);
    if ( (_DWORD)result )
    {
      v8 = *(unsigned int *)(a1 + 136);
      if ( (unsigned int)v8 <= 0x40 )
      {
LABEL_8:
        result = *(_QWORD *)(a1 + 120);
        for ( j = result + 16 * v8; j != result; result += 16 )
          *(_QWORD *)result = -4096;
        goto LABEL_10;
      }
      result = sub_C7D6A0(*(_QWORD *)(a1 + 120), 16LL * (unsigned int)v8, 8);
      *(_DWORD *)(a1 + 136) = 0;
LABEL_25:
      *(_QWORD *)(a1 + 120) = 0;
LABEL_10:
      *(_QWORD *)(a1 + 128) = 0;
    }
  }
LABEL_11:
  *(_DWORD *)(a1 + 104) = 0;
  return result;
}
