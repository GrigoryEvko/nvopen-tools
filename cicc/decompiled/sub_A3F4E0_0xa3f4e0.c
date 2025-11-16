// Function: sub_A3F4E0
// Address: 0xa3f4e0
//
__int64 __fastcall sub_A3F4E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r9d
  int v4; // r8d
  _QWORD *v5; // r8
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  unsigned __int8 v8; // dl
  __int64 *v9; // rbx
  __int64 v10; // rcx
  __int64 *v11; // r13
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  _BYTE *v16; // r10
  int v17; // edx
  _BYTE *v18; // r8
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  int v22; // r11d
  _BYTE *v23; // [rsp+0h] [rbp-250h]
  _QWORD *v24; // [rsp+10h] [rbp-240h] BYREF
  __int64 v25; // [rsp+18h] [rbp-238h]
  _QWORD v26[70]; // [rsp+20h] [rbp-230h] BYREF

  result = 0x4000000000LL;
  v3 = *(_DWORD *)(a2 + 8);
  v24 = v26;
  v25 = 0x4000000000LL;
  if ( v3 )
  {
    v4 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = 0;
    if ( v4 )
    {
      result = *(_QWORD *)a2;
      if ( (unsigned __int8)(**(_BYTE **)a2 - 5) <= 0x1Fu )
      {
        LODWORD(v25) = 1;
        v5 = v26;
        v26[0] = result;
        LODWORD(result) = 1;
        do
        {
          v6 = (unsigned int)result;
          result = (unsigned int)(result - 1);
          v7 = v5[v6 - 1];
          LODWORD(v25) = result;
          v8 = *(_BYTE *)(v7 - 16);
          if ( (v8 & 2) != 0 )
          {
            v9 = *(__int64 **)(v7 - 32);
            v10 = *(unsigned int *)(v7 - 24);
          }
          else
          {
            v10 = (*(_WORD *)(v7 - 16) >> 6) & 0xF;
            v9 = (__int64 *)(v7 + -16 - 8LL * ((v8 >> 2) & 0xF));
          }
          v11 = &v9[v10];
          if ( v9 != v11 )
          {
            do
            {
              v12 = *v9;
              if ( *v9 )
              {
                v13 = *(unsigned int *)(a1 + 280);
                v14 = *(_QWORD *)(a1 + 264);
                if ( (_DWORD)v13 )
                {
                  v7 = ((_DWORD)v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
                  v15 = v14 + 16 * v7;
                  v16 = *(_BYTE **)v15;
                  if ( v12 == *(_QWORD *)v15 )
                  {
LABEL_11:
                    if ( v15 != v14 + 16 * v13 )
                    {
                      if ( *(_DWORD *)(v15 + 8) )
                      {
                        v17 = *(_DWORD *)(v15 + 12);
                        *(_DWORD *)(v15 + 8) = 0;
                        if ( v17 )
                        {
                          v18 = *(_BYTE **)v15;
                          if ( (unsigned __int8)(**(_BYTE **)v15 - 5) <= 0x1Fu )
                          {
                            v19 = (unsigned int)v25;
                            v20 = (unsigned int)v25 + 1LL;
                            if ( v20 > HIDWORD(v25) )
                            {
                              v7 = (unsigned __int64)v26;
                              v23 = v18;
                              sub_C8D5F0(&v24, v26, v20, 8);
                              v19 = (unsigned int)v25;
                              v18 = v23;
                            }
                            v24[v19] = v18;
                            LODWORD(v25) = v25 + 1;
                          }
                        }
                      }
                    }
                  }
                  else
                  {
                    v21 = 1;
                    while ( v16 != (_BYTE *)-4096LL )
                    {
                      v22 = v21 + 1;
                      v7 = ((_DWORD)v13 - 1) & (unsigned int)(v21 + v7);
                      v15 = v14 + 16LL * (unsigned int)v7;
                      v16 = *(_BYTE **)v15;
                      if ( v12 == *(_QWORD *)v15 )
                        goto LABEL_11;
                      v21 = v22;
                    }
                  }
                }
              }
              ++v9;
            }
            while ( v11 != v9 );
            result = (unsigned int)v25;
            v5 = v24;
          }
        }
        while ( (_DWORD)result );
        if ( v5 != v26 )
          return _libc_free(v5, v7);
      }
    }
  }
  return result;
}
