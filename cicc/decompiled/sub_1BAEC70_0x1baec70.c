// Function: sub_1BAEC70
// Address: 0x1baec70
//
void __fastcall sub_1BAEC70(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  __int64 v6; // r8
  int v7; // r11d
  __int64 *v8; // r10
  unsigned int v9; // edx
  __int64 *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned int v13; // esi
  int v14; // ecx
  int v15; // edi
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 v18; // [rsp-48h] [rbp-48h] BYREF
  _QWORD v19[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 != a3 )
  {
    v5 = a2;
    while ( 1 )
    {
      v12 = *v5;
      v13 = *(_DWORD *)(a1 + 24);
      v18 = *v5;
      if ( !v13 )
        break;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = 1;
      v8 = 0;
      v9 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = (__int64 *)(v6 + 8LL * v9);
      v11 = *v10;
      if ( v12 == *v10 )
      {
LABEL_4:
        v5 += 3;
        if ( a3 == v5 )
          return;
      }
      else
      {
        while ( v11 != -8 )
        {
          if ( v8 || v11 != -16 )
            v10 = v8;
          v9 = (v13 - 1) & (v7 + v9);
          v11 = *(_QWORD *)(v6 + 8LL * v9);
          if ( v12 == v11 )
            goto LABEL_4;
          ++v7;
          v8 = v10;
          v10 = (__int64 *)(v6 + 8LL * v9);
        }
        if ( !v8 )
          v8 = v10;
        v15 = *(_DWORD *)(a1 + 16);
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
        *v8 = v12;
        v16 = *v5;
        v17 = *(_BYTE **)(a1 + 40);
        v19[0] = *v5;
        if ( v17 == *(_BYTE **)(a1 + 48) )
        {
          sub_12879C0(a1 + 32, v17, v19);
          goto LABEL_4;
        }
        if ( v17 )
        {
          *(_QWORD *)v17 = v16;
          v17 = *(_BYTE **)(a1 + 40);
        }
        v5 += 3;
        *(_QWORD *)(a1 + 40) = v17 + 8;
        if ( a3 == v5 )
          return;
      }
    }
    ++*(_QWORD *)a1;
LABEL_7:
    v13 *= 2;
LABEL_8:
    sub_1353F00(a1, v13);
    sub_1A97120(a1, &v18, v19);
    v8 = (__int64 *)v19[0];
    v12 = v18;
    v14 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_18;
  }
}
