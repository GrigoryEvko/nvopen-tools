// Function: sub_315EB70
// Address: 0x315eb70
//
__int64 __fastcall sub_315EB70(__int64 a1, __int64 **a2)
{
  __int64 v2; // r14
  __int64 v4; // rcx
  __int64 v5; // r13
  __int64 v6; // rcx
  unsigned __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r15
  bool v11; // cc
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r12
  unsigned __int64 v17; // rdi
  __int64 v19; // [rsp+0h] [rbp-70h]
  __int64 v20; // [rsp+8h] [rbp-68h]
  int v21; // [rsp+10h] [rbp-60h]
  char v22; // [rsp+17h] [rbp-59h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  _QWORD v24[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = a1;
  v4 = *(_QWORD *)(a1 + 16);
  ++*(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v5 = *(_QWORD *)a1;
  v20 = v4;
  LODWORD(v4) = *(_DWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 32) = 0;
  v21 = v4;
  v6 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 24) = 0;
  v19 = v6;
  v22 = *(_BYTE *)(a1 + 40);
  while ( 1 )
  {
    sub_B4CED0((__int64)v24, v5, **a2);
    v7 = v24[0];
    sub_B4CED0((__int64)v24, *(_QWORD *)(v2 - 48), **a2);
    if ( v24[0] >= v7 )
      break;
    v8 = *(unsigned int *)(v2 + 32);
    *(_QWORD *)v2 = *(_QWORD *)(v2 - 48);
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD *)(v2 + 16);
      v10 = v9 + 32 * v8;
      do
      {
        while ( 1 )
        {
          if ( *(_QWORD *)v9 != -8192 && *(_QWORD *)v9 != -4096 )
          {
            if ( *(_BYTE *)(v9 + 24) )
            {
              v11 = *(_DWORD *)(v9 + 16) <= 0x40u;
              *(_BYTE *)(v9 + 24) = 0;
              if ( !v11 )
              {
                v12 = *(_QWORD *)(v9 + 8);
                if ( v12 )
                  break;
              }
            }
          }
          v9 += 32;
          if ( v10 == v9 )
            goto LABEL_12;
        }
        v23 = v9;
        j_j___libc_free_0_0(v12);
        v9 = v23 + 32;
      }
      while ( v10 != v23 + 32 );
LABEL_12:
      v8 = *(unsigned int *)(v2 + 32);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 16), 32 * v8, 8);
    v13 = *(_QWORD *)(v2 - 32);
    ++*(_QWORD *)(v2 + 8);
    ++*(_QWORD *)(v2 - 40);
    v2 -= 48;
    *(_QWORD *)(v2 + 64) = v13;
    LODWORD(v13) = *(_DWORD *)(v2 + 24);
    *(_QWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 72) = v13;
    LODWORD(v13) = *(_DWORD *)(v2 + 28);
    *(_DWORD *)(v2 + 24) = 0;
    *(_DWORD *)(v2 + 76) = v13;
    LODWORD(v13) = *(_DWORD *)(v2 + 32);
    *(_DWORD *)(v2 + 28) = 0;
    *(_DWORD *)(v2 + 80) = v13;
    LOBYTE(v13) = *(_BYTE *)(v2 + 40);
    *(_DWORD *)(v2 + 32) = 0;
    *(_BYTE *)(v2 + 88) = v13;
  }
  v14 = *(unsigned int *)(v2 + 32);
  *(_QWORD *)v2 = v5;
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD *)(v2 + 16);
    v16 = v15 + 32 * v14;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v15 != -8192 && *(_QWORD *)v15 != -4096 )
        {
          if ( *(_BYTE *)(v15 + 24) )
          {
            v11 = *(_DWORD *)(v15 + 16) <= 0x40u;
            *(_BYTE *)(v15 + 24) = 0;
            if ( !v11 )
            {
              v17 = *(_QWORD *)(v15 + 8);
              if ( v17 )
                break;
            }
          }
        }
        v15 += 32;
        if ( v16 == v15 )
          goto LABEL_23;
      }
      j_j___libc_free_0_0(v17);
      v15 += 32;
    }
    while ( v16 != v15 );
LABEL_23:
    LODWORD(v14) = *(_DWORD *)(v2 + 32);
  }
  sub_C7D6A0(*(_QWORD *)(v2 + 16), 32LL * (unsigned int)v14, 8);
  ++*(_QWORD *)(v2 + 8);
  *(_QWORD *)(v2 + 16) = v20;
  *(_QWORD *)(v2 + 24) = v19;
  *(_DWORD *)(v2 + 32) = v21;
  *(_BYTE *)(v2 + 40) = v22;
  return sub_C7D6A0(0, 0, 8);
}
