// Function: sub_1949270
// Address: 0x1949270
//
__int64 __fastcall sub_1949270(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  unsigned int v14; // r10d
  __int64 v15; // rcx
  _QWORD *v16; // rsi
  __int64 v17; // r11
  unsigned __int64 v18; // r8
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 result; // rax

  v6 = sub_157F280(*(_QWORD *)(a1 + 8));
  v7 = 0;
  v9 = v8;
  v10 = v6;
  while ( v9 != v10 )
  {
    if ( (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 0 )
    {
      v11 = 0;
      v12 = 8LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      do
      {
        while ( 1 )
        {
          v13 = (*(_BYTE *)(v10 + 23) & 0x40) != 0
              ? *(_QWORD *)(v10 - 8)
              : v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
          if ( *(_QWORD *)(v11 + v13 + 24LL * *(unsigned int *)(v10 + 56) + 8) == a2 )
          {
            v14 = v7 + 1;
            v15 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v7);
            v16 = (_QWORD *)(3 * v11 + v13);
            if ( *v16 )
            {
              v17 = v16[1];
              v18 = v16[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v18 = v17;
              if ( v17 )
                *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
            }
            *v16 = v15;
            v7 = v14;
            if ( v15 )
              break;
          }
          v11 += 8;
          if ( v12 == v11 )
            goto LABEL_16;
        }
        v19 = *(_QWORD *)(v15 + 8);
        v16[1] = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v19 + 16) & 3LL;
        v11 += 8;
        v16[2] = (v15 + 8) | v16[2] & 3LL;
        v7 = v14;
        *(_QWORD *)(v15 + 8) = v16;
      }
      while ( v12 != v11 );
    }
LABEL_16:
    v20 = *(_QWORD *)(v10 + 32);
    if ( !v20 )
      BUG();
    v10 = 0;
    if ( *(_BYTE *)(v20 - 8) == 77 )
      v10 = v20 - 24;
  }
  result = *(_QWORD *)(a3 + 40);
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
