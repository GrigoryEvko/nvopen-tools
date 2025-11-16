// Function: sub_11A3D00
// Address: 0x11a3d00
//
_BYTE *__fastcall sub_11A3D00(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r11
  __int64 v11; // r10
  unsigned int v12; // r15d
  _BYTE *result; // rax
  _BYTE *v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-38h]

  v7 = a5;
  v8 = a3;
  if ( *(_BYTE *)a2 == 85
    && (v23 = *(_QWORD *)(a2 - 32)) != 0
    && !*(_BYTE *)v23
    && *(_QWORD *)(v23 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v23 + 33) & 0x20) != 0 )
  {
    v10 = *(_QWORD *)(a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  }
  else
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v9 = *(_QWORD *)(a2 - 8);
    else
      v9 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v10 = *(_QWORD *)(v9 + 32 * v8);
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = **(_DWORD **)a1 + 1;
  v28 = *(_DWORD *)(a4 + 8);
  if ( v28 > 0x40 )
  {
    v25 = v10;
    v26 = v11;
    sub_C43780((__int64)&v27, (const void **)a4);
    v7 = a5;
    v10 = v25;
    v11 = v26;
  }
  else
  {
    v27 = *(_QWORD *)a4;
  }
  result = (_BYTE *)sub_11A3F30(v11, v10, &v27, v7, v12, 0);
  v14 = result;
  if ( v28 > 0x40 && v27 )
    result = (_BYTE *)j_j___libc_free_0_0(v27);
  if ( v14 )
  {
    v15 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v16 = 32 * v8 + *(_QWORD *)(a2 - 8);
    else
      v16 = a2 + 32 * (v8 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v17 = *(_QWORD *)v16;
    if ( *(_QWORD *)v16 )
    {
      v18 = *(_QWORD *)(v16 + 8);
      **(_QWORD **)(v16 + 16) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v16 + 16);
    }
    *(_QWORD *)v16 = v14;
    v19 = *((_QWORD *)v14 + 2);
    *(_QWORD *)(v16 + 8) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = v16 + 8;
    *(_QWORD *)(v16 + 16) = v14 + 16;
    *((_QWORD *)v14 + 2) = v16;
    if ( *(_BYTE *)v17 > 0x1Cu )
    {
      v20 = *(_QWORD *)(v15 + 40);
      v27 = v17;
      v21 = v20 + 2096;
      sub_11A2F60(v21, &v27);
      v22 = *(_QWORD *)(v17 + 16);
      if ( v22 )
      {
        if ( !*(_QWORD *)(v22 + 8) )
        {
          v27 = *(_QWORD *)(v22 + 24);
          sub_11A2F60(v21, &v27);
        }
      }
    }
    result = *(_BYTE **)(a1 + 16);
    *result = 1;
  }
  return result;
}
