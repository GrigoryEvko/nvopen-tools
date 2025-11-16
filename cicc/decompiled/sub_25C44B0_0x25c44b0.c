// Function: sub_25C44B0
// Address: 0x25c44b0
//
void __fastcall sub_25C44B0(__int64 a1, unsigned int a2)
{
  char v2; // al
  __int64 *v3; // r12
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // ebx
  __int64 v8; // rbx
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // r8
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  unsigned int v20; // [rsp+14h] [rbp-1A4Ch]
  __int64 v22; // [rsp+20h] [rbp-1A40h]
  __int64 v23; // [rsp+28h] [rbp-1A38h]
  __int64 v24[838]; // [rsp+30h] [rbp-1A30h] BYREF

  v2 = *(_BYTE *)(a1 + 8);
  v3 = *(__int64 **)(a1 + 16);
  v20 = a2;
  v4 = v2 & 1;
  if ( a2 <= 0x10 )
  {
    if ( !v4 )
    {
      *(_BYTE *)(a1 + 8) = v2 | 1;
      v7 = *(_DWORD *)(a1 + 24);
      goto LABEL_6;
    }
    v8 = a1 + 16;
    v22 = a1 + 6672;
  }
  else
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v20 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v8 = a1 + 16;
      v22 = a1 + 6672;
      if ( !v4 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v6 = 416LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = 26624;
        v20 = 64;
        v7 = *(_DWORD *)(a1 + 24);
LABEL_5:
        *(_QWORD *)(a1 + 16) = sub_C7D670(v6, 8);
        *(_DWORD *)(a1 + 24) = v20;
LABEL_6:
        sub_25C41C0(a1, v3, &v3[52 * v7]);
        sub_C7D6A0((__int64)v3, 416LL * v7, 8);
        return;
      }
      v8 = a1 + 16;
      v20 = 64;
      v22 = a1 + 6672;
    }
  }
  v9 = v24;
  do
  {
    v10 = *(_QWORD *)v8;
    if ( *(_QWORD *)v8 != -4096 && v10 != -8192 )
    {
      if ( v9 )
        *v9 = v10;
      v9[1] = 0;
      v9[2] = 1;
      v11 = v9 + 3;
      do
      {
        if ( v11 )
          *v11 = -4096;
        v11 += 12;
      }
      while ( v11 != v9 + 51 );
      sub_25C3E80((__int64)(v9 + 1), v8 + 8);
      v9 += 52;
      *((_BYTE *)v9 - 8) = *(_BYTE *)(v8 + 408);
      *((_BYTE *)v9 - 7) = *(_BYTE *)(v8 + 409);
      if ( (*(_BYTE *)(v8 + 16) & 1) != 0 )
      {
        v13 = v8 + 24;
        v23 = v8 + 408;
      }
      else
      {
        v12 = *(unsigned int *)(v8 + 32);
        v13 = *(_QWORD *)(v8 + 24);
        v14 = 96 * v12;
        if ( !(_DWORD)v12 )
          goto LABEL_42;
        v23 = v13 + v14;
        if ( v13 == v13 + v14 )
          goto LABEL_42;
      }
      do
      {
        if ( *(_QWORD *)v13 != -4096 && *(_QWORD *)v13 != -8192 )
        {
          v15 = *(_QWORD *)(v13 + 16);
          v16 = 32LL * *(unsigned int *)(v13 + 24);
          v17 = v15 + v16;
          if ( v15 != v15 + v16 )
          {
            do
            {
              v17 -= 32LL;
              if ( *(_DWORD *)(v17 + 24) > 0x40u )
              {
                v18 = *(_QWORD *)(v17 + 16);
                if ( v18 )
                  j_j___libc_free_0_0(v18);
              }
              if ( *(_DWORD *)(v17 + 8) > 0x40u && *(_QWORD *)v17 )
                j_j___libc_free_0_0(*(_QWORD *)v17);
            }
            while ( v15 != v17 );
            v17 = *(_QWORD *)(v13 + 16);
          }
          if ( v17 != v13 + 32 )
            _libc_free(v17);
        }
        v13 += 96;
      }
      while ( v23 != v13 );
      if ( (*(_BYTE *)(v8 + 16) & 1) == 0 )
      {
        v13 = *(_QWORD *)(v8 + 24);
        v14 = 96LL * *(unsigned int *)(v8 + 32);
LABEL_42:
        sub_C7D6A0(v13, v14, 8);
      }
    }
    v8 += 416;
  }
  while ( v8 != v22 );
  if ( v20 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v19 = sub_C7D670(416LL * v20, 8);
    *(_DWORD *)(a1 + 24) = v20;
    *(_QWORD *)(a1 + 16) = v19;
  }
  sub_25C41C0(a1, v24, v9);
}
