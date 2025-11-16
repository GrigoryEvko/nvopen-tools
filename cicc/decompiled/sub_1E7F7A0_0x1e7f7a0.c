// Function: sub_1E7F7A0
// Address: 0x1e7f7a0
//
__int64 __fastcall sub_1E7F7A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned int v7; // r8d
  const void *v8; // rsi
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned __int64 v12; // r10
  __int64 v13; // rdx
  _QWORD *v14; // rdx
  __int64 v16; // [rsp+0h] [rbp-50h]
  unsigned __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int8 v18; // [rsp+17h] [rbp-39h]

  v3 = *(_QWORD *)(a1 + 32);
  v4 = v3 + 40LL * *(unsigned int *)(a1 + 40);
  if ( v3 == v4 )
  {
    return 0;
  }
  else
  {
    v7 = 0;
    v8 = (const void *)(a2 + 16);
    do
    {
      if ( !*(_BYTE *)v3 )
      {
        v9 = *(_DWORD *)(v3 + 8);
        if ( v9 )
        {
          if ( v9 > 0 )
          {
            v7 = 1;
          }
          else if ( (*(_BYTE *)(v3 + 4) & 1) == 0
                 && (*(_BYTE *)(v3 + 4) & 2) == 0
                 && ((*(_BYTE *)(v3 + 3) & 0x10) == 0 || (*(_DWORD *)v3 & 0xFFF00) != 0) )
          {
            v10 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 16LL * (v9 & 0x7FFFFFFF) + 8);
            if ( v10 )
            {
              if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
              {
                v10 = *(_QWORD *)(v10 + 32);
                if ( v10 )
                {
                  if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
                    BUG();
                }
              }
            }
            v11 = *(_QWORD *)(v10 + 16);
            v12 = ((unsigned __int64)(-858993459 * (unsigned int)((v3 - *(_QWORD *)(a1 + 32)) >> 3)) << 32)
                | (-858993459 * (unsigned int)((v10 - *(_QWORD *)(v11 + 32)) >> 3));
            v13 = *(unsigned int *)(a2 + 8);
            if ( (unsigned int)v13 >= *(_DWORD *)(a2 + 12) )
            {
              v16 = *(_QWORD *)(v10 + 16);
              v17 = ((unsigned __int64)(-858993459 * (unsigned int)((v3 - *(_QWORD *)(a1 + 32)) >> 3)) << 32)
                  | (-858993459 * (unsigned int)((v10 - *(_QWORD *)(v11 + 32)) >> 3));
              v18 = v7;
              sub_16CD150(a2, v8, 0, 16, v7, -858993459);
              v13 = *(unsigned int *)(a2 + 8);
              v11 = v16;
              v12 = v17;
              v7 = v18;
            }
            v14 = (_QWORD *)(*(_QWORD *)a2 + 16 * v13);
            *v14 = v11;
            v14[1] = v12;
            ++*(_DWORD *)(a2 + 8);
          }
        }
      }
      v3 += 40;
    }
    while ( v4 != v3 );
  }
  return v7;
}
