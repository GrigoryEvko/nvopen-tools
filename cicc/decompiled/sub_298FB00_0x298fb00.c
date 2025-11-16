// Function: sub_298FB00
// Address: 0x298fb00
//
void __fastcall sub_298FB00(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v4; // r13
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // eax
  _QWORD *v9; // r9
  __int64 v10; // r8
  char v11; // cl
  unsigned int v12; // esi
  unsigned int v13; // eax
  int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdx
  int v17; // r11d
  _QWORD *v18; // r10
  __int64 *v19; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 != a3 )
  {
    v4 = a2;
    while ( 1 )
    {
      v11 = *(_BYTE *)(a1 + 8) & 1;
      if ( v11 )
      {
        v6 = a1 + 16;
        v7 = 3;
      }
      else
      {
        v12 = *(_DWORD *)(a1 + 24);
        v6 = *(_QWORD *)(a1 + 16);
        if ( !v12 )
        {
          v13 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v19 = 0;
          v14 = (v13 >> 1) + 1;
          goto LABEL_10;
        }
        v7 = v12 - 1;
      }
      v8 = v7 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v9 = (_QWORD *)(v6 + 8LL * v8);
      v10 = *v9;
      if ( *v4 == *v9 )
      {
LABEL_5:
        if ( a3 == ++v4 )
          return;
      }
      else
      {
        v17 = 1;
        v18 = 0;
        while ( v10 != -4096 )
        {
          if ( v10 != -8192 || v18 )
            v9 = v18;
          v8 = v7 & (v17 + v8);
          v10 = *(_QWORD *)(v6 + 8LL * v8);
          if ( *v4 == v10 )
            goto LABEL_5;
          ++v17;
          v18 = v9;
          v9 = (_QWORD *)(v6 + 8LL * v8);
        }
        v13 = *(_DWORD *)(a1 + 8);
        if ( !v18 )
          v18 = v9;
        ++*(_QWORD *)a1;
        v19 = v18;
        v14 = (v13 >> 1) + 1;
        if ( !v11 )
        {
          v12 = *(_DWORD *)(a1 + 24);
LABEL_10:
          if ( 3 * v12 <= 4 * v14 )
            goto LABEL_22;
          goto LABEL_11;
        }
        v12 = 4;
        if ( (unsigned int)(4 * v14) >= 0xC )
        {
LABEL_22:
          v12 *= 2;
LABEL_23:
          sub_298F720(a1, v12);
          sub_298C0C0(a1, v4, &v19);
          v13 = *(_DWORD *)(a1 + 8);
          goto LABEL_12;
        }
LABEL_11:
        if ( v12 - *(_DWORD *)(a1 + 12) - v14 <= v12 >> 3 )
          goto LABEL_23;
LABEL_12:
        *(_DWORD *)(a1 + 8) = (2 * (v13 >> 1) + 2) | v13 & 1;
        v15 = v19;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a1 + 12);
        v16 = *v4++;
        *v15 = v16;
        if ( a3 == v4 )
          return;
      }
    }
  }
}
