// Function: sub_1B2A5B0
// Address: 0x1b2a5b0
//
void __fastcall sub_1B2A5B0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v13; // rdx
  __int64 i; // r12
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // ecx
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 *v22; // rax
  __int64 v23; // r10
  __int64 v24; // rcx
  int v25; // eax
  int v26; // r9d
  _QWORD *v27; // [rsp+8h] [rbp-38h]

  v10 = a2 + 72;
  v11 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v11 )
  {
    if ( !v11 )
      BUG();
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 24);
      if ( v13 != v11 + 16 )
        break;
      v11 = *(_QWORD *)(v11 + 8);
      if ( v10 == v11 )
        return;
      if ( !v11 )
        BUG();
    }
    if ( v10 != v11 )
    {
      while ( 1 )
      {
        for ( i = *(_QWORD *)(v13 + 8); ; i = *(_QWORD *)(v11 + 24) )
        {
          v15 = v11 - 24;
          if ( !v11 )
            v15 = 0;
          if ( i != v15 + 40 )
            break;
          v11 = *(_QWORD *)(v11 + 8);
          if ( v10 == v11 )
            break;
          if ( !v11 )
            BUG();
        }
        v16 = *(_DWORD *)(a1 + 104);
        v17 = v13 - 24;
        v18 = 0;
        if ( v16 )
        {
          v19 = v16 - 1;
          v20 = *(_QWORD *)(a1 + 88);
          v21 = v19 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v17 == *v22 )
          {
LABEL_18:
            v18 = v22[1];
          }
          else
          {
            v25 = 1;
            while ( v23 != -8 )
            {
              v26 = v25 + 1;
              v21 = v19 & (v25 + v21);
              v22 = (__int64 *)(v20 + 16LL * v21);
              v23 = *v22;
              if ( v17 == *v22 )
                goto LABEL_18;
              v25 = v26;
            }
            v18 = 0;
          }
        }
        if ( *(_BYTE *)(v13 - 8) == 78 )
        {
          v24 = *(_QWORD *)(v13 - 48);
          if ( !*(_BYTE *)(v24 + 16) && (*(_BYTE *)(v24 + 33) & 0x20) != 0 && v18 && *(_DWORD *)(v24 + 36) == 197 )
          {
            v27 = (_QWORD *)(v13 - 24);
            sub_164D160(
              v17,
              *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v13 - 4) & 0xFFFFFFF)),
              a3,
              a4,
              a5,
              a6,
              a7,
              a8,
              a9,
              a10);
            sub_15F20C0(v27);
          }
        }
        if ( v11 == v10 )
          break;
        v13 = i;
      }
    }
  }
}
