// Function: sub_18F3C00
// Address: 0x18f3c00
//
__int64 __fastcall sub_18F3C00(__int64 a1, _QWORD **a2)
{
  _QWORD *v3; // r12
  _QWORD *v4; // r15
  char v5; // al
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  unsigned __int8 v9; // cl
  __int64 result; // rax
  __int64 v11; // rdx
  int v12; // ecx
  __int64 v13; // r9
  int v14; // r8d
  unsigned int v15; // ecx
  _QWORD *v16; // rdi
  _QWORD *v17; // r10
  unsigned int v18; // ecx
  int v19; // edi
  int v20; // r11d
  __int64 v21; // [rsp+0h] [rbp-80h]
  __int64 v22; // [rsp+8h] [rbp-78h]
  unsigned __int8 v23; // [rsp+1Fh] [rbp-61h]
  _QWORD v24[12]; // [rsp+20h] [rbp-60h] BYREF

  v3 = *a2;
  v4 = **(_QWORD ***)a1;
  v22 = *(_QWORD *)(a1 + 16);
  v21 = **(_QWORD **)(a1 + 24);
  v23 = sub_15E4690(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL), 0);
  v5 = sub_140E950(v3, v24, v22, v21, v23 << 16);
  v6 = -1;
  if ( v5 )
    v6 = v24[0];
  v7 = 0;
  v8 = **(_QWORD **)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 > 0x17u )
  {
    if ( v9 == 78 )
    {
      v7 = v8 | 4;
    }
    else if ( v9 == 29 )
    {
      v7 = **(_QWORD **)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  v24[1] = v6;
  v24[0] = v3;
  memset(&v24[2], 0, 24);
  result = sub_134F0E0(v4, v7, (__int64)v24) & 1;
  if ( (_DWORD)result )
  {
    v11 = *(_QWORD *)(a1 + 40);
    if ( (*(_BYTE *)(v11 + 8) & 1) != 0 )
    {
      v13 = v11 + 16;
      v14 = 15;
    }
    else
    {
      v12 = *(_DWORD *)(v11 + 24);
      v13 = *(_QWORD *)(v11 + 16);
      if ( !v12 )
        return result;
      v14 = v12 - 1;
    }
    v15 = v14 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v16 = (_QWORD *)(v13 + 8LL * v15);
    v17 = (_QWORD *)*v16;
    if ( *a2 == (_QWORD *)*v16 )
    {
LABEL_14:
      *v16 = -16;
      v18 = *(_DWORD *)(v11 + 8);
      ++*(_DWORD *)(v11 + 12);
      *(_DWORD *)(v11 + 8) = (2 * (v18 >> 1) - 2) | v18 & 1;
    }
    else
    {
      v19 = 1;
      while ( v17 != (_QWORD *)-8LL )
      {
        v20 = v19 + 1;
        v15 = v14 & (v19 + v15);
        v16 = (_QWORD *)(v13 + 8LL * v15);
        v17 = (_QWORD *)*v16;
        if ( *a2 == (_QWORD *)*v16 )
          goto LABEL_14;
        v19 = v20;
      }
    }
  }
  return result;
}
