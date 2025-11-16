// Function: sub_2EB4EE0
// Address: 0x2eb4ee0
//
char __fastcall sub_2EB4EE0(__int64 a1, __int64 a2)
{
  char v3; // al
  _QWORD *v4; // r13
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned __int64 v7; // r9
  __int64 v8; // r12
  unsigned __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rbx
  void *v12; // rdi
  __int64 v13; // r8
  size_t v14; // rdx
  unsigned int v15; // r13d
  __int64 v16; // rdi
  __int64 v17; // rax
  int v19; // [rsp+4h] [rbp-3Ch]
  int v20; // [rsp+4h] [rbp-3Ch]
  unsigned __int64 v21; // [rsp+8h] [rbp-38h]
  unsigned __int64 v22; // [rsp+8h] [rbp-38h]

  sub_2EB46D0(a1);
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 72LL * *(unsigned int *)(a1 + 24), 8);
  v3 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v3;
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 && *(_DWORD *)(a2 + 24) > 4u )
  {
    *(_BYTE *)(a1 + 8) = v3 & 0xFE;
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v16 = 288;
      v15 = 4;
    }
    else
    {
      v15 = *(_DWORD *)(a2 + 24);
      v16 = 72LL * v15;
    }
    v17 = sub_C7D670(v16, 8);
    *(_DWORD *)(a1 + 24) = v15;
    *(_QWORD *)(a1 + 16) = v17;
  }
  v4 = (_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE | *(_DWORD *)(a1 + 8) & 1;
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  LOBYTE(v5) = *(_BYTE *)(a1 + 8) & 1;
  if ( !(_BYTE)v5 )
    v4 = *(_QWORD **)(a1 + 16);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v6 = a2 + 16;
    if ( !(_BYTE)v5 )
      goto LABEL_9;
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 16);
    if ( !(_BYTE)v5 )
    {
LABEL_9:
      v7 = *(unsigned int *)(a1 + 24);
      if ( !*(_DWORD *)(a1 + 24) )
        return v5;
      goto LABEL_10;
    }
  }
  v7 = 4;
LABEL_10:
  v8 = v6 + 72;
  v9 = 0;
  do
  {
    if ( v4 )
    {
      v5 = *(_QWORD *)(v8 - 72);
      *v4 = v5;
    }
    else
    {
      v5 = MEMORY[0];
    }
    if ( v5 != -8192 )
    {
      v10 = v8 - 64;
      v11 = (__int64)(v4 + 1);
      LOBYTE(v5) = v5 != -4096;
      if ( (_BYTE)v5 )
      {
        do
        {
          v12 = (void *)(v11 + 16);
          *(_DWORD *)(v11 + 8) = 0;
          *(_QWORD *)v11 = v11 + 16;
          *(_DWORD *)(v11 + 12) = 2;
          v13 = *(unsigned int *)(v10 + 8);
          if ( v10 != v11 && (_DWORD)v13 )
          {
            v14 = 8LL * (unsigned int)v13;
            if ( (unsigned int)v13 <= 2
              || (v20 = *(_DWORD *)(v10 + 8),
                  v22 = v7,
                  LOBYTE(v5) = sub_C8D5F0(v11, (const void *)(v11 + 16), (unsigned int)v13, 8u, v13, v7),
                  v12 = *(void **)v11,
                  v7 = v22,
                  LODWORD(v13) = v20,
                  (v14 = 8LL * *(unsigned int *)(v10 + 8)) != 0) )
            {
              v19 = v13;
              v21 = v7;
              LOBYTE(v5) = (unsigned __int8)memcpy(v12, *(const void **)v10, v14);
              LODWORD(v13) = v19;
              v7 = v21;
            }
            *(_DWORD *)(v11 + 8) = v13;
          }
          v10 += 32;
          v11 += 32;
        }
        while ( v10 != v8 );
      }
    }
    ++v9;
    v4 += 9;
    v8 += 72;
  }
  while ( v9 < v7 );
  return v5;
}
