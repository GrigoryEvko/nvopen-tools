// Function: sub_28CB5E0
// Address: 0x28cb5e0
//
__int64 __fastcall sub_28CB5E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v5; // r15
  __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 *v13; // rdx
  unsigned int v14; // ebx
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned int v19; // r8d
  __int64 *v20; // rcx
  __int64 v21; // r10
  int v22; // ecx
  int v23; // r9d
  _QWORD v24[4]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v25[10]; // [rsp+20h] [rbp-50h] BYREF

  v3 = *(unsigned int *)(a2 + 84);
  if ( *(_DWORD *)(a2 + 84) - *(_DWORD *)(a2 + 88) == 1 || *(_QWORD *)(a1 + 1392) == a2 )
  {
    v7 = *(_QWORD *)(a2 + 72);
    if ( !*(_BYTE *)(a2 + 92) )
      v3 = *(unsigned int *)(a2 + 80);
    v25[0] = *(_QWORD *)(a2 + 72);
    v25[1] = v7 + 8 * v3;
    sub_254BBF0((__int64)v25);
    v25[2] = a2 + 64;
    v25[3] = *(_QWORD *)(a2 + 64);
    return *(_QWORD *)v25[0];
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 24);
    if ( !v5 )
    {
      v8 = *(_QWORD **)(a2 + 72);
      v9 = a2 + 64;
      if ( !*(_BYTE *)(a2 + 92) )
        v3 = *(unsigned int *)(a2 + 80);
      v10 = &v8[v3];
      v24[0] = *(_QWORD *)(a2 + 72);
      for ( v24[1] = v10; v8 != v10; v24[0] = v8 )
      {
        if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v8;
      }
      v11 = *(_QWORD *)(a2 + 64);
      v24[2] = v9;
      v24[3] = v11;
      sub_28CB5A0(v25, v9);
      v12 = v25[0];
      v13 = (__int64 *)v24[0];
      if ( v24[0] != v25[0] )
      {
        v14 = -1;
        do
        {
          v15 = *(_DWORD *)(a1 + 2440);
          v16 = *v13;
          v17 = *(_QWORD *)(a1 + 2424);
          if ( v15 )
          {
            v18 = v15 - 1;
            v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v20 = (__int64 *)(v17 + 16LL * v19);
            v21 = *v20;
            if ( v16 == *v20 )
            {
LABEL_17:
              v15 = *((_DWORD *)v20 + 2);
            }
            else
            {
              v22 = 1;
              while ( v21 != -4096 )
              {
                v23 = v22 + 1;
                v19 = v18 & (v22 + v19);
                v20 = (__int64 *)(v17 + 16LL * v19);
                v21 = *v20;
                if ( v16 == *v20 )
                  goto LABEL_17;
                v22 = v23;
              }
              v15 = 0;
            }
          }
          if ( v15 < v14 )
          {
            v14 = v15;
            v5 = *v13;
          }
          v24[0] = v13 + 1;
          sub_254BBF0((__int64)v24);
          v13 = (__int64 *)v24[0];
        }
        while ( v24[0] != v12 );
      }
    }
  }
  return v5;
}
