// Function: sub_153E970
// Address: 0x153e970
//
void __fastcall sub_153E970(__int64 a1, __int64 a2)
{
  int v2; // r9d
  int v3; // r8d
  _BYTE *v4; // rax
  _QWORD *v6; // rdi
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // rax
  _BYTE *v17; // r9
  int v18; // edx
  _BYTE *v19; // r8
  __int64 v20; // rax
  int v21; // eax
  int v22; // r10d
  _BYTE *v23; // [rsp+8h] [rbp-248h]
  _QWORD *v24; // [rsp+10h] [rbp-240h] BYREF
  __int64 v25; // [rsp+18h] [rbp-238h]
  _QWORD v26[70]; // [rsp+20h] [rbp-230h] BYREF

  v2 = *(_DWORD *)(a2 + 8);
  v24 = v26;
  v25 = 0x4000000000LL;
  if ( v2 )
  {
    v3 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = 0;
    if ( v3 )
    {
      v4 = *(_BYTE **)a2;
      if ( (unsigned __int8)(**(_BYTE **)a2 - 4) <= 0x1Eu )
      {
        LODWORD(v25) = 1;
        v6 = v26;
        v26[0] = v4;
        v7 = 1;
        do
        {
          v8 = v7--;
          v9 = v6[v8 - 1];
          LODWORD(v25) = v7;
          v10 = 8LL * *(unsigned int *)(v9 + 8);
          v11 = (__int64 *)(v9 - v10);
          if ( v9 - v10 != v9 )
          {
            do
            {
              v12 = *v11;
              if ( *v11 )
              {
                v13 = *(unsigned int *)(a1 + 280);
                if ( (_DWORD)v13 )
                {
                  v14 = *(_QWORD *)(a1 + 264);
                  v15 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
                  v16 = v14 + 16LL * v15;
                  v17 = *(_BYTE **)v16;
                  if ( v12 == *(_QWORD *)v16 )
                  {
LABEL_9:
                    if ( v16 != v14 + 16 * v13 )
                    {
                      if ( *(_DWORD *)(v16 + 8) )
                      {
                        v18 = *(_DWORD *)(v16 + 12);
                        *(_DWORD *)(v16 + 8) = 0;
                        if ( v18 )
                        {
                          v19 = *(_BYTE **)v16;
                          if ( (unsigned __int8)(**(_BYTE **)v16 - 4) <= 0x1Eu )
                          {
                            v20 = (unsigned int)v25;
                            if ( (unsigned int)v25 >= HIDWORD(v25) )
                            {
                              v23 = v19;
                              sub_16CD150(&v24, v26, 0, 8);
                              v20 = (unsigned int)v25;
                              v19 = v23;
                            }
                            v24[v20] = v19;
                            LODWORD(v25) = v25 + 1;
                          }
                        }
                      }
                    }
                  }
                  else
                  {
                    v21 = 1;
                    while ( v17 != (_BYTE *)-4LL )
                    {
                      v22 = v21 + 1;
                      v15 = (v13 - 1) & (v21 + v15);
                      v16 = v14 + 16LL * v15;
                      v17 = *(_BYTE **)v16;
                      if ( v12 == *(_QWORD *)v16 )
                        goto LABEL_9;
                      v21 = v22;
                    }
                  }
                }
              }
              ++v11;
            }
            while ( (__int64 *)v9 != v11 );
            v7 = v25;
            v6 = v24;
          }
        }
        while ( v7 );
        if ( v6 != v26 )
          _libc_free((unsigned __int64)v6);
      }
    }
  }
}
