// Function: sub_28D33B0
// Address: 0x28d33b0
//
void __fastcall sub_28D33B0(__int64 a1, _BYTE *a2, __int64 *a3)
{
  __int64 v3; // r8
  __int64 v6; // r13
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // r11d
  __int64 v11; // rcx
  __int64 *v12; // rsi
  __int64 *v13; // rdx
  __int64 v14; // r10
  __int64 v15; // rsi
  int v16; // ecx
  int v17; // esi
  _BYTE *v18; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v19[10]; // [rsp+10h] [rbp-50h] BYREF

  v18 = a2;
  if ( *a2 > 0x1Cu )
  {
    v3 = *(unsigned int *)(a1 + 1720);
    v6 = a1 + 1696;
    if ( (_DWORD)v3 )
    {
      v7 = (unsigned int)(v3 - 1);
      v8 = (__int64)a2;
      v9 = *(_QWORD *)(a1 + 1704);
      v10 = 1;
      v11 = (unsigned int)v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (__int64 *)(v9 + 56 * v11);
      v13 = 0;
      v14 = *v12;
      if ( v8 == *v12 )
      {
LABEL_4:
        v15 = (__int64)(v12 + 1);
LABEL_5:
        sub_2411830((__int64)v19, v15, a3, v11, v3, v7);
        return;
      }
      while ( v14 != -4096 )
      {
        if ( !v13 && v14 == -8192 )
          v13 = v12;
        v11 = (unsigned int)v7 & (v10 + (_DWORD)v11);
        v12 = (__int64 *)(v9 + 56LL * (unsigned int)v11);
        v14 = *v12;
        if ( v8 == *v12 )
          goto LABEL_4;
        ++v10;
      }
      v16 = *(_DWORD *)(a1 + 1712);
      if ( !v13 )
        v13 = v12;
      ++*(_QWORD *)(a1 + 1696);
      v17 = v16 + 1;
      v19[0] = v13;
      if ( 4 * (v16 + 1) < (unsigned int)(3 * v3) )
      {
        v11 = (unsigned int)(v3 - *(_DWORD *)(a1 + 1716) - v17);
        if ( (unsigned int)v11 > (unsigned int)v3 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a1 + 1712) = v17;
          if ( *v13 != -4096 )
            --*(_DWORD *)(a1 + 1716);
          *v13 = v8;
          v15 = (__int64)(v13 + 1);
          v13[1] = 0;
          v13[2] = (__int64)(v13 + 5);
          v13[3] = 2;
          *((_DWORD *)v13 + 8) = 0;
          *((_BYTE *)v13 + 36) = 1;
          goto LABEL_5;
        }
        sub_28D3190(v6, v3);
LABEL_21:
        sub_28CBAD0(v6, (__int64 *)&v18, v19);
        v11 = *(unsigned int *)(a1 + 1712);
        v8 = (__int64)v18;
        v13 = (__int64 *)v19[0];
        v17 = v11 + 1;
        goto LABEL_16;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 1696);
      v19[0] = 0;
    }
    sub_28D3190(v6, 2 * v3);
    goto LABEL_21;
  }
}
