// Function: sub_3574DA0
// Address: 0x3574da0
//
__int64 __fastcall sub_3574DA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  char v9; // al
  int v10; // esi
  unsigned __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 *v16; // r9
  int v17; // ecx
  unsigned int v18; // edx
  __int64 *v19; // rdi
  __int64 v20; // r8
  int v22; // edi
  int v23; // r10d
  __int64 v24[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a2 + 32);
  v6 = v5 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  while ( v6 != v5 )
  {
    if ( !*(_BYTE *)v5 )
    {
      v9 = *(_BYTE *)(v5 + 4);
      if ( (v9 & 1) == 0 && (v9 & 2) == 0 && ((*(_BYTE *)(v5 + 3) & 0x10) == 0 || (*(_DWORD *)v5 & 0xFFF00) != 0) )
      {
        v10 = *(_DWORD *)(v5 + 8);
        if ( (unsigned int)(v10 - 1) <= 0x3FFFFFFE )
          return 1;
        v11 = sub_2EBEE10(*(_QWORD *)(*(_QWORD *)(a1 + 216) + 32LL), v10);
        v12 = *(_DWORD *)(a3 + 72);
        v13 = *(_QWORD *)(v11 + 24);
        v24[0] = v13;
        if ( v12 )
        {
          v14 = *(_QWORD *)(a3 + 64);
          v15 = *(unsigned int *)(a3 + 80);
          v16 = (__int64 *)(v14 + 8 * v15);
          if ( (_DWORD)v15 )
          {
            v17 = v15 - 1;
            v18 = v17 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v19 = (__int64 *)(v14 + 8LL * v18);
            v20 = *v19;
            if ( v13 == *v19 )
            {
LABEL_14:
              if ( v16 != v19 )
                return 1;
            }
            else
            {
              v22 = 1;
              while ( v20 != -4096 )
              {
                v23 = v22 + 1;
                v18 = v17 & (v22 + v18);
                v19 = (__int64 *)(v14 + 8LL * v18);
                v20 = *v19;
                if ( v13 == *v19 )
                  goto LABEL_14;
                v22 = v23;
              }
            }
          }
        }
        else
        {
          v7 = *(_QWORD **)(a3 + 88);
          v8 = &v7[*(unsigned int *)(a3 + 96)];
          if ( v8 != sub_3574250(v7, (__int64)v8, v24) )
            return 1;
        }
      }
    }
    v5 += 40;
  }
  return 0;
}
