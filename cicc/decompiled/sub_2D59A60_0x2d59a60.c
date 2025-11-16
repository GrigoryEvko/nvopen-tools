// Function: sub_2D59A60
// Address: 0x2d59a60
//
__int64 __fastcall sub_2D59A60(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // edi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // r14
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r15
  int v21; // edi
  __int64 v22; // rcx
  __int64 v23; // rsi
  int v24; // edi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  int v29; // eax
  int v30; // r9d
  int v31; // eax
  int v32; // r9d
  __int64 v33; // [rsp+0h] [rbp-40h] BYREF
  __int64 v34[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(_DWORD *)(a3 + 24);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(a3 + 8);
  if ( v5 )
  {
    v8 = v5 - 1;
    v10 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
    {
LABEL_3:
      v13 = v11[1];
      if ( v13 && **(_QWORD **)(v13 + 32) == v6 && sub_D47930(v13) )
      {
        v15 = sub_D47930(v13);
        v16 = *(_QWORD *)(a2 - 8);
        v17 = v15;
        if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
        {
          v18 = 0;
          while ( v17 != *(_QWORD *)(v16 + 32LL * *(unsigned int *)(a2 + 72) + 8 * v18) )
          {
            if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == (_DWORD)++v18 )
              goto LABEL_24;
          }
          v19 = 32 * v18;
        }
        else
        {
LABEL_24:
          v19 = 0x1FFFFFFFE0LL;
        }
        v20 = *(_QWORD *)(v16 + v19);
        if ( *(_BYTE *)v20 > 0x1Cu )
        {
          v21 = *(_DWORD *)(a3 + 24);
          v22 = *(_QWORD *)(v20 + 40);
          v23 = *(_QWORD *)(a3 + 8);
          if ( v21 )
          {
            v24 = v21 - 1;
            v25 = v24 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v26 = (__int64 *)(v23 + 16LL * v25);
            v27 = *v26;
            if ( v22 == *v26 )
            {
LABEL_16:
              if ( v13 == v26[1] )
              {
                v33 = 0;
                v34[0] = 0;
                if ( (unsigned __int8)sub_2D57670((char *)v20, &v33, v34) )
                {
                  if ( v33 == a2 )
                  {
                    v28 = v34[0];
                    *(_QWORD *)a1 = v20;
                    *(_BYTE *)(a1 + 16) = 1;
                    *(_QWORD *)(a1 + 8) = v28;
                    return a1;
                  }
                }
              }
            }
            else
            {
              v31 = 1;
              while ( v27 != -4096 )
              {
                v32 = v31 + 1;
                v25 = v24 & (v31 + v25);
                v26 = (__int64 *)(v23 + 16LL * v25);
                v27 = *v26;
                if ( v22 == *v26 )
                  goto LABEL_16;
                v31 = v32;
              }
            }
          }
        }
      }
    }
    else
    {
      v29 = 1;
      while ( v12 != -4096 )
      {
        v30 = v29 + 1;
        v10 = v8 & (v29 + v10);
        v11 = (__int64 *)(v7 + 16LL * v10);
        v12 = *v11;
        if ( v6 == *v11 )
          goto LABEL_3;
        v29 = v30;
      }
    }
  }
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
