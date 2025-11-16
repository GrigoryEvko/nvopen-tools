// Function: sub_2C35B20
// Address: 0x2c35b20
//
__int64 __fastcall sub_2C35B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rdx
  _DWORD *v9; // rax
  _DWORD *i; // rdx
  __int64 v11; // rax
  __int64 result; // rax
  unsigned int v13; // ecx
  unsigned int v14; // eax
  int v15; // r13d
  unsigned int v16; // eax
  int v17; // eax
  __int64 v18; // r9
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rdx
  int v22; // eax
  __int64 v23; // r8
  __int64 v24; // r12
  __int64 v25; // [rsp+8h] [rbp-38h] BYREF
  __int64 v26; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v27[5]; // [rsp+18h] [rbp-28h] BYREF

  v6 = a1 + 48;
  v25 = a2;
  v7 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  if ( !v7 )
  {
    if ( !*(_DWORD *)(a1 + 68) )
    {
LABEL_7:
      *(_DWORD *)(a1 + 88) = 0;
LABEL_8:
      v11 = 0;
      if ( !*(_DWORD *)(a1 + 92) )
      {
        sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), 1u, 8u, a5, a6);
        v11 = 8LL * *(unsigned int *)(a1 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 80) + v11) = v25;
      result = (unsigned int)(*(_DWORD *)(a1 + 88) + 1);
      *(_DWORD *)(a1 + 88) = result;
      if ( (unsigned int)result > 2 )
        return sub_2AD96C0(v6);
      return result;
    }
    v8 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v8 <= 0x40 )
      goto LABEL_4;
    sub_C7D6A0(*(_QWORD *)(a1 + 56), 8 * v8, 4);
    *(_DWORD *)(a1 + 72) = 0;
LABEL_32:
    *(_QWORD *)(a1 + 56) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 64) = 0;
    goto LABEL_7;
  }
  v13 = 4 * v7;
  v8 = *(unsigned int *)(a1 + 72);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v8 <= v13 )
  {
LABEL_4:
    v9 = *(_DWORD **)(a1 + 56);
    for ( i = &v9[2 * v8]; i != v9; *((_BYTE *)v9 - 4) = 1 )
    {
      *v9 = -1;
      v9 += 2;
    }
    goto LABEL_6;
  }
  v14 = v7 - 1;
  if ( v14 )
  {
    _BitScanReverse(&v14, v14);
    v15 = 1 << (33 - (v14 ^ 0x1F));
    if ( v15 < 64 )
      v15 = 64;
    if ( v15 == (_DWORD)v8 )
      goto LABEL_21;
  }
  else
  {
    v15 = 64;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 8 * v8, 4);
  v16 = sub_2C261E0(v15);
  *(_DWORD *)(a1 + 72) = v16;
  if ( !v16 )
    goto LABEL_32;
  *(_QWORD *)(a1 + 56) = sub_C7D670(8LL * v16, 4);
LABEL_21:
  sub_2C2BF80(v6);
  v17 = *(_DWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 88) = 0;
  if ( !v17 )
    goto LABEL_8;
  result = sub_2AC3BB0(v6, (int *)&v25, &v26);
  if ( (_BYTE)result )
    return result;
  v19 = *(_DWORD *)(a1 + 72);
  v20 = *(_DWORD *)(a1 + 64);
  v21 = v26;
  ++*(_QWORD *)(a1 + 48);
  v22 = v20 + 1;
  v23 = 2 * v19;
  v27[0] = v21;
  if ( 4 * v22 >= 3 * v19 )
  {
    v19 *= 2;
  }
  else if ( v19 - *(_DWORD *)(a1 + 68) - v22 > v19 >> 3 )
  {
    goto LABEL_25;
  }
  sub_2AD9490(v6, v19);
  sub_2AC3BB0(v6, (int *)&v25, v27);
  v21 = v27[0];
  v22 = *(_DWORD *)(a1 + 64) + 1;
LABEL_25:
  *(_DWORD *)(a1 + 64) = v22;
  if ( *(_DWORD *)v21 != -1 || !*(_BYTE *)(v21 + 4) )
    --*(_DWORD *)(a1 + 68);
  *(_DWORD *)v21 = v25;
  *(_BYTE *)(v21 + 4) = BYTE4(v25);
  result = *(unsigned int *)(a1 + 88);
  v24 = v25;
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), result + 1, 8u, v23, v18);
    result = *(unsigned int *)(a1 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * result) = v24;
  ++*(_DWORD *)(a1 + 88);
  return result;
}
