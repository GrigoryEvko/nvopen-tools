// Function: sub_1E51930
// Address: 0x1e51930
//
void __fastcall sub_1E51930(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 *v5; // r12
  __int64 v7; // r8
  __int64 *v8; // r10
  int v9; // r11d
  unsigned int v10; // eax
  __int64 *v11; // rdi
  __int64 v12; // rcx
  unsigned int v13; // esi
  int v14; // edx
  int v15; // eax
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 *v18; // [rsp-40h] [rbp-40h] BYREF

  if ( a3 != a2 )
  {
    v4 = a1 + 32;
    v5 = a2;
    while ( 1 )
    {
      v13 = *(_DWORD *)(a1 + 24);
      if ( !v13 )
        break;
      v7 = *(_QWORD *)(a1 + 8);
      v8 = 0;
      v9 = 1;
      v10 = (v13 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
      v11 = (__int64 *)(v7 + 8LL * v10);
      v12 = *v11;
      if ( *v5 == *v11 )
      {
LABEL_4:
        if ( a3 == ++v5 )
          return;
      }
      else
      {
        while ( v12 != -8 )
        {
          if ( v12 != -16 || v8 )
            v11 = v8;
          v10 = (v13 - 1) & (v9 + v10);
          v12 = *(_QWORD *)(v7 + 8LL * v10);
          if ( *v5 == v12 )
            goto LABEL_4;
          ++v9;
          v8 = v11;
          v11 = (__int64 *)(v7 + 8LL * v10);
        }
        v15 = *(_DWORD *)(a1 + 16);
        if ( !v8 )
          v8 = v11;
        ++*(_QWORD *)a1;
        v14 = v15 + 1;
        if ( 4 * (v15 + 1) >= 3 * v13 )
          goto LABEL_7;
        if ( v13 - *(_DWORD *)(a1 + 20) - v14 <= v13 >> 3 )
          goto LABEL_8;
LABEL_18:
        *(_DWORD *)(a1 + 16) = v14;
        if ( *v8 != -8 )
          --*(_DWORD *)(a1 + 20);
        v16 = *v5;
        *v8 = *v5;
        v17 = *(_BYTE **)(a1 + 40);
        if ( v17 == *(_BYTE **)(a1 + 48) )
        {
          sub_1CFD630(v4, v17, v5);
          goto LABEL_4;
        }
        if ( v17 )
        {
          *(_QWORD *)v17 = v16;
          v17 = *(_BYTE **)(a1 + 40);
        }
        ++v5;
        *(_QWORD *)(a1 + 40) = v17 + 8;
        if ( a3 == v5 )
          return;
      }
    }
    ++*(_QWORD *)a1;
LABEL_7:
    v13 *= 2;
LABEL_8:
    sub_1E512C0(a1, v13);
    sub_1E49160(a1, v5, &v18);
    v8 = v18;
    v14 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_18;
  }
}
