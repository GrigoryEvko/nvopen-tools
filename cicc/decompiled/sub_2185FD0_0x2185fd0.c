// Function: sub_2185FD0
// Address: 0x2185fd0
//
__int64 __fastcall sub_2185FD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 result; // rax
  unsigned int v7; // ebx
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edx
  _DWORD *v11; // rdi
  int v12; // ecx
  int v13; // edx
  unsigned int v14; // ebx
  unsigned int v15; // r8d
  unsigned int v16; // esi
  int v17; // ebx
  __int64 v18; // r9
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // r10
  __int64 v21; // rcx
  unsigned __int64 v22; // rdi
  int v25; // r11d
  _DWORD *v26; // r10
  int v27; // edi
  int v28; // ecx
  unsigned int v29; // [rsp+14h] [rbp-3Ch] BYREF
  _DWORD *v30; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(_QWORD *)(sub_1C01EA0(a1, a2) + 8);
  result = sub_217D950((__int64 *)v5, 0, *(_DWORD *)(v5 + 16));
  if ( (_DWORD)result != -1 )
  {
    v7 = result;
    while ( 1 )
    {
      result = sub_2185E80(a1, v7);
      v8 = *(_DWORD *)(a3 + 24);
      v29 = result;
      if ( !v8 )
        break;
      v9 = *(_QWORD *)(a3 + 8);
      v10 = (v8 - 1) & (37 * result);
      v11 = (_DWORD *)(v9 + 4LL * v10);
      v12 = *v11;
      if ( (_DWORD)result == *v11 )
        goto LABEL_5;
      v25 = 1;
      v26 = 0;
      while ( v12 != -1 )
      {
        if ( v26 || v12 != -2 )
          v11 = v26;
        v10 = (v8 - 1) & (v25 + v10);
        v12 = *(_DWORD *)(v9 + 4LL * v10);
        if ( (_DWORD)result == v12 )
          goto LABEL_5;
        ++v25;
        v26 = v11;
        v11 = (_DWORD *)(v9 + 4LL * v10);
      }
      if ( !v26 )
        v26 = v11;
      v27 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v28 = v27 + 1;
      if ( 4 * (v27 + 1) >= 3 * v8 )
        goto LABEL_28;
      if ( v8 - *(_DWORD *)(a3 + 20) - v28 <= v8 >> 3 )
        goto LABEL_29;
LABEL_24:
      *(_DWORD *)(a3 + 16) = v28;
      if ( *v26 != -1 )
        --*(_DWORD *)(a3 + 20);
      *v26 = result;
LABEL_5:
      v13 = *(_DWORD *)(v5 + 16);
      v14 = v7 + 1;
      if ( v13 != v14 )
      {
        v15 = v14 >> 6;
        v16 = (unsigned int)(v13 - 1) >> 6;
        if ( v14 >> 6 <= v16 )
        {
          v17 = v14 & 0x3F;
          v18 = *(_QWORD *)v5;
          v19 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17);
          if ( v17 == 0 )
            v19 = 0;
          v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
          v21 = v15;
          v22 = ~v19;
          while ( 1 )
          {
            _RDX = *(_QWORD *)(v18 + 8 * v21);
            if ( v15 == (_DWORD)v21 )
              _RDX = v22 & *(_QWORD *)(v18 + 8 * v21);
            result = v20 & _RDX;
            if ( v16 == (_DWORD)v21 )
              _RDX &= v20;
            if ( _RDX )
              break;
            if ( v16 < (unsigned int)++v21 )
              return result;
          }
          __asm { tzcnt   rdx, rdx }
          v7 = _RDX + ((_DWORD)v21 << 6);
          if ( v7 != -1 )
            continue;
        }
      }
      return result;
    }
    ++*(_QWORD *)a3;
LABEL_28:
    v8 *= 2;
LABEL_29:
    sub_136B240(a3, v8);
    sub_1DF91F0(a3, (int *)&v29, &v30);
    v26 = v30;
    result = v29;
    v28 = *(_DWORD *)(a3 + 16) + 1;
    goto LABEL_24;
  }
  return result;
}
