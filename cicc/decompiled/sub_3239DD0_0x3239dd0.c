// Function: sub_3239DD0
// Address: 0x3239dd0
//
__int64 __fastcall sub_3239DD0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rbx
  int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // ecx
  _QWORD *v17; // rax
  _QWORD *j; // rdx
  unsigned int v19; // eax
  _QWORD *v20; // rdi
  __int64 v21; // rbx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *i; // rdx
  _QWORD *v27; // rax

  *(_QWORD *)(a1 + 3048) = a2;
  v4 = sub_B92180(*a2);
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
  {
    result = *(_QWORD *)(v4 - 32);
    v7 = *(_QWORD *)(result + 40);
    if ( !*(_DWORD *)(v7 + 32) )
      return result;
LABEL_5:
    v8 = sub_3238860(a1, v7);
    v11 = (unsigned int)sub_3737600(v8);
    v12 = 0;
    if ( (_BYTE)v11 )
      v12 = sub_E9E970(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL), v7, v9, v10, v11);
    *(_QWORD *)(a1 + 3672) = v12;
    v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL);
    *(_DWORD *)(v13 + 1912) = sub_321F7C0(a1, v8);
    v14 = *(_DWORD *)(a1 + 6248);
    ++*(_QWORD *)(a1 + 6232);
    if ( v14 )
    {
      v16 = 4 * v14;
      v15 = *(unsigned int *)(a1 + 6256);
      if ( (unsigned int)(4 * v14) < 0x40 )
        v16 = 64;
      if ( (unsigned int)v15 <= v16 )
        goto LABEL_15;
      v19 = v14 - 1;
      if ( v19 )
      {
        _BitScanReverse(&v19, v19);
        v20 = *(_QWORD **)(a1 + 6240);
        v21 = (unsigned int)(1 << (33 - (v19 ^ 0x1F)));
        if ( (int)v21 < 64 )
          v21 = 64;
        if ( (_DWORD)v21 == (_DWORD)v15 )
        {
          *(_QWORD *)(a1 + 6248) = 0;
          v27 = &v20[v21];
          do
          {
            if ( v20 )
              *v20 = -4096;
            ++v20;
          }
          while ( v27 != v20 );
          goto LABEL_11;
        }
      }
      else
      {
        v20 = *(_QWORD **)(a1 + 6240);
        LODWORD(v21) = 64;
      }
      sub_C7D6A0((__int64)v20, 8 * v15, 8);
      v22 = ((((((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v21 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v21 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 16;
      v23 = (v22
           | (((((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v21 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v21 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 6256) = v23;
      v24 = (_QWORD *)sub_C7D670(8 * v23, 8);
      v25 = *(unsigned int *)(a1 + 6256);
      *(_QWORD *)(a1 + 6248) = 0;
      *(_QWORD *)(a1 + 6240) = v24;
      for ( i = &v24[v25]; i != v24; ++v24 )
      {
        if ( v24 )
          *v24 = -4096;
      }
    }
    else if ( *(_DWORD *)(a1 + 6252) )
    {
      v15 = *(unsigned int *)(a1 + 6256);
      if ( (unsigned int)v15 > 0x40 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 6240), 8 * v15, 8);
        *(_QWORD *)(a1 + 6240) = 0;
        *(_QWORD *)(a1 + 6248) = 0;
        *(_DWORD *)(a1 + 6256) = 0;
        goto LABEL_11;
      }
LABEL_15:
      v17 = *(_QWORD **)(a1 + 6240);
      for ( j = &v17[v15]; j != v17; ++v17 )
        *v17 = -4096;
      *(_QWORD *)(a1 + 6248) = 0;
    }
LABEL_11:
    sub_322B4B0(a1);
    *(_QWORD *)(a1 + 48) = sub_3239A30(
                             a1,
                             a2,
                             *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL) + 1912LL));
    return sub_322C940(a1, (__int64)a2);
  }
  result = v4 - 16 - 8LL * ((v5 >> 2) & 0xF);
  v7 = *(_QWORD *)(result + 40);
  if ( *(_DWORD *)(v7 + 32) )
    goto LABEL_5;
  return result;
}
