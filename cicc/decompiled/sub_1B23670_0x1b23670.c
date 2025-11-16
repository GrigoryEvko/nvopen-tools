// Function: sub_1B23670
// Address: 0x1b23670
//
void __fastcall sub_1B23670(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // r15
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r13
  unsigned int v9; // eax
  int v10; // esi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rdx
  int v14; // r12d
  unsigned int v15; // r12d
  int v16; // r14d
  __int64 v17; // rdi
  __int64 v18; // rsi
  _BYTE *v19; // r12
  _BYTE *v20; // r14
  unsigned int v21; // esi
  unsigned int v23; // [rsp+18h] [rbp-78h]
  __int64 i; // [rsp+20h] [rbp-70h]
  _BYTE *v26; // [rsp+30h] [rbp-60h] BYREF
  __int64 v27; // [rsp+38h] [rbp-58h]
  _BYTE v28[80]; // [rsp+40h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a1 + 48);
  for ( i = sub_157ED20(a1) + 24; i != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    if ( !v5 )
      BUG();
    v8 = v5 - 24;
    v9 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
    if ( v9 )
    {
      v10 = 0;
      v11 = 24LL * *(unsigned int *)(v5 + 32) + 8;
      while ( 1 )
      {
        v12 = v8 - 24LL * v9;
        if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
          v12 = *(_QWORD *)(v5 - 32);
        v13 = (_QWORD *)(v11 + v12);
        v14 = v10 + 1;
        if ( a2 == *v13 )
          break;
        v11 += 8;
        if ( v9 == v14 )
          goto LABEL_11;
        ++v10;
      }
      v14 = v10;
      *v13 = a3;
LABEL_11:
      v15 = v14 + 1;
      v26 = v28;
      v27 = 0x800000000LL;
      if ( v15 >= v9 || !a4 )
        continue;
      v16 = a4;
      v17 = 0;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
          v18 = *(_QWORD *)(v5 - 32);
        else
          v18 = v8 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF);
        if ( a2 == *(_QWORD *)(v18 + 8LL * v15 + 24LL * *(unsigned int *)(v5 + 32) + 8) )
        {
          if ( HIDWORD(v27) <= (unsigned int)v17 )
          {
            v23 = v9;
            sub_16CD150((__int64)&v26, v28, 0, 4, v6, v7);
            v17 = (unsigned int)v27;
            v9 = v23;
          }
          --v16;
          *(_DWORD *)&v26[4 * v17] = v15++;
          v17 = (unsigned int)(v27 + 1);
          LODWORD(v27) = v27 + 1;
          if ( v15 >= v9 )
          {
LABEL_23:
            v19 = v26;
            v20 = &v26[4 * v17];
            if ( v26 != v20 )
            {
              do
              {
                v21 = *((_DWORD *)v20 - 1);
                v20 -= 4;
                sub_15F5350(v5 - 24, v21, 1);
              }
              while ( v19 != v20 );
              v20 = v26;
            }
            if ( v20 != v28 )
              _libc_free((unsigned __int64)v20);
            break;
          }
        }
        else if ( ++v15 >= v9 )
        {
          goto LABEL_23;
        }
        if ( !v16 )
          goto LABEL_23;
      }
    }
  }
}
