// Function: sub_2E5EC00
// Address: 0x2e5ec00
//
__int64 __fastcall sub_2E5EC00(__int64 a1)
{
  __int64 v1; // r10
  __int64 v2; // rax
  __int64 *v3; // r8
  __int64 *v4; // r11
  int v5; // ebx
  __int64 v6; // r13
  __int64 v7; // r9
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  bool v10; // al
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 *v14; // rsi
  int v15; // edx
  unsigned int v16; // eax
  __int64 *v17; // rdi
  __int64 v18; // r14
  int v19; // edi
  int v20; // r15d
  __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( *(_DWORD *)(a1 + 16) == 1 )
  {
    v1 = a1;
    v2 = **(_QWORD **)(a1 + 8);
    v3 = *(__int64 **)(v2 + 64);
    v4 = &v3[*(unsigned int *)(v2 + 72)];
    if ( v3 != v4 )
    {
      v5 = *(_DWORD *)(a1 + 72);
      v6 = 0;
      while ( 1 )
      {
        v7 = *v3;
        v21[0] = *v3;
        if ( !v5 )
          break;
        v12 = *(_QWORD *)(v1 + 64);
        v13 = *(unsigned int *)(v1 + 80);
        v14 = (__int64 *)(v12 + 8 * v13);
        if ( !(_DWORD)v13 )
          goto LABEL_8;
        v15 = v13 - 1;
        v16 = v15 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v17 = (__int64 *)(v12 + 8LL * v16);
        v18 = *v17;
        if ( v7 == *v17 )
        {
LABEL_14:
          v10 = v14 != v17;
LABEL_7:
          if ( !v10 )
            goto LABEL_8;
          if ( v4 == ++v3 )
            return v6;
        }
        else
        {
          v19 = 1;
          while ( v18 != -4096 )
          {
            v20 = v19 + 1;
            v16 = v15 & (v19 + v16);
            v17 = (__int64 *)(v12 + 8LL * v16);
            v18 = *v17;
            if ( v7 == *v17 )
              goto LABEL_14;
            v19 = v20;
          }
LABEL_8:
          if ( v6 && v7 != v6 )
            return 0;
          ++v3;
          v6 = v7;
          if ( v4 == v3 )
            return v6;
        }
      }
      v8 = *(_QWORD **)(v1 + 88);
      v9 = &v8[*(unsigned int *)(v1 + 96)];
      v10 = v9 != sub_2E5D7F0(v8, (__int64)v9, v21);
      goto LABEL_7;
    }
  }
  return 0;
}
