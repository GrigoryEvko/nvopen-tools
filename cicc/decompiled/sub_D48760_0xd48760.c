// Function: sub_D48760
// Address: 0xd48760
//
__int64 __fastcall sub_D48760(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rax
  int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // esi
  int v13; // eax
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned int v21; // ebx
  int v22; // eax
  bool v23; // al
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h] BYREF
  __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = **(_QWORD **)(a1 + 32);
  v27 = 0;
  v28[0] = 0;
  if ( (unsigned __int8)sub_D48630(a1, &v27, v28) )
  {
    v4 = *(_QWORD *)(v3 + 56);
    v5 = v27;
    v6 = v28[0];
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      if ( *(_BYTE *)(v4 - 24) != 84 )
        break;
      if ( !a2 || *(_QWORD *)(v4 - 16) == a2 )
      {
        v7 = *(_QWORD *)(v4 - 32);
        v8 = 0x1FFFFFFFE0LL;
        v9 = *(_DWORD *)(v4 - 20) & 0x7FFFFFF;
        if ( v9 )
        {
          v10 = 0;
          do
          {
            if ( v5 == *(_QWORD *)(v7 + 32LL * *(unsigned int *)(v4 + 48) + 8 * v10) )
            {
              v8 = 32 * v10;
              goto LABEL_13;
            }
            ++v10;
          }
          while ( v9 != (_DWORD)v10 );
          v8 = 0x1FFFFFFFE0LL;
        }
LABEL_13:
        v11 = *(_QWORD *)(v7 + v8);
        if ( *(_BYTE *)v11 == 17 )
        {
          v12 = *(_DWORD *)(v11 + 32);
          if ( v12 <= 0x40 )
          {
            v14 = *(_QWORD *)(v11 + 24) == 0;
          }
          else
          {
            v25 = v6;
            v13 = sub_C444A0(v11 + 24);
            v6 = v25;
            v14 = v12 == v13;
          }
          if ( v14 )
          {
            v15 = 0x1FFFFFFFE0LL;
            if ( v9 )
            {
              v16 = 0;
              do
              {
                if ( v6 == *(_QWORD *)(v7 + 32LL * *(unsigned int *)(v4 + 48) + 8 * v16) )
                {
                  v15 = 32 * v16;
                  goto LABEL_22;
                }
                ++v16;
              }
              while ( v9 != (_DWORD)v16 );
              v15 = 0x1FFFFFFFE0LL;
            }
LABEL_22:
            v17 = *(_QWORD *)(v7 + v15);
            if ( *(_BYTE *)v17 == 42 )
            {
              v18 = (*(_BYTE *)(v17 + 7) & 0x40) != 0
                  ? *(_QWORD **)(v17 - 8)
                  : (_QWORD *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
              v19 = v4 - 24;
              if ( v4 - 24 == *v18 )
              {
                v20 = v18[4];
                if ( *(_BYTE *)v20 == 17 )
                {
                  v21 = *(_DWORD *)(v20 + 32);
                  if ( v21 <= 0x40 )
                  {
                    v23 = *(_QWORD *)(v20 + 24) == 1;
                  }
                  else
                  {
                    v26 = v6;
                    v22 = sub_C444A0(v20 + 24);
                    v6 = v26;
                    v23 = v21 - 1 == v22;
                  }
                  if ( v23 )
                    return v19;
                }
              }
            }
          }
        }
      }
      v4 = *(_QWORD *)(v4 + 8);
    }
  }
  return 0;
}
