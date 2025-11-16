// Function: sub_30D1740
// Address: 0x30d1740
//
__int64 __fastcall sub_30D1740(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r10
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // edx
  unsigned int v11; // eax
  __int64 v12; // r8
  int v13; // edi
  int v15; // edx
  int v16; // r11d

  v2 = *(unsigned int *)(a1 + 192);
  v3 = *(_QWORD *)(a1 + 176);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 16 * v2) )
      {
        v7 = *(_DWORD *)(a1 + 224);
        v8 = v5[1];
        v9 = *(_QWORD *)(a1 + 208);
        if ( v7 )
        {
          v10 = v7 - 1;
          v11 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v12 = *(_QWORD *)(v9 + 8LL * v11);
          v13 = 1;
          if ( v8 == v12 )
            return v12;
          while ( v12 != -4096 )
          {
            v11 = v10 & (v13 + v11);
            v12 = *(_QWORD *)(v9 + 8LL * v11);
            if ( v8 == v12 )
              return v12;
            ++v13;
          }
        }
      }
    }
    else
    {
      v15 = 1;
      while ( v6 != -4096 )
      {
        v16 = v15 + 1;
        v4 = (v2 - 1) & (v15 + v4);
        v5 = (__int64 *)(v3 + 16LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v15 = v16;
      }
    }
  }
  return 0;
}
