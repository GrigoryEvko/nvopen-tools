// Function: sub_1A97AC0
// Address: 0x1a97ac0
//
void __fastcall sub_1A97AC0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v3; // r12
  __int64 v5; // r13
  __int64 v6; // r8
  __int64 *v7; // r10
  int v8; // r11d
  unsigned int v9; // eax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  unsigned int v12; // esi
  int v13; // edx
  int v14; // eax
  __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 *v17; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(__int64 **)(a2 + 40);
  v3 = *(__int64 **)(a2 + 32);
  if ( v3 != v2 )
  {
    v5 = a1 + 32;
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 24);
      if ( !v12 )
        break;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = 0;
      v8 = 1;
      v9 = (v12 - 1) & (((unsigned int)*v3 >> 9) ^ ((unsigned int)*v3 >> 4));
      v10 = (__int64 *)(v6 + 8LL * v9);
      v11 = *v10;
      if ( *v3 == *v10 )
      {
LABEL_4:
        if ( v2 == ++v3 )
          return;
      }
      else
      {
        while ( v11 != -8 )
        {
          if ( v7 || v11 != -16 )
            v10 = v7;
          v9 = (v12 - 1) & (v8 + v9);
          v11 = *(_QWORD *)(v6 + 8LL * v9);
          if ( *v3 == v11 )
            goto LABEL_4;
          ++v8;
          v7 = v10;
          v10 = (__int64 *)(v6 + 8LL * v9);
        }
        v14 = *(_DWORD *)(a1 + 16);
        if ( !v7 )
          v7 = v10;
        ++*(_QWORD *)a1;
        v13 = v14 + 1;
        if ( 4 * (v14 + 1) >= 3 * v12 )
          goto LABEL_7;
        if ( v12 - *(_DWORD *)(a1 + 20) - v13 <= v12 >> 3 )
          goto LABEL_8;
LABEL_18:
        *(_DWORD *)(a1 + 16) = v13;
        if ( *v7 != -8 )
          --*(_DWORD *)(a1 + 20);
        v15 = *v3;
        *v7 = *v3;
        v16 = *(_BYTE **)(a1 + 40);
        if ( v16 == *(_BYTE **)(a1 + 48) )
        {
          sub_1287830(v5, v16, v3);
          goto LABEL_4;
        }
        if ( v16 )
        {
          *(_QWORD *)v16 = v15;
          v16 = *(_BYTE **)(a1 + 40);
        }
        ++v3;
        *(_QWORD *)(a1 + 40) = v16 + 8;
        if ( v2 == v3 )
          return;
      }
    }
    ++*(_QWORD *)a1;
LABEL_7:
    v12 *= 2;
LABEL_8:
    sub_1353F00(a1, v12);
    sub_1A97120(a1, v3, &v17);
    v7 = v17;
    v13 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_18;
  }
}
