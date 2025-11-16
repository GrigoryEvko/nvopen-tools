// Function: sub_2FBF8B0
// Address: 0x2fbf8b0
//
__int64 __fastcall sub_2FBF8B0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  int v6; // edx
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  char v12; // dl
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rbx
  int v16; // r14d
  __int64 v17; // rax
  char v18; // r10
  __int64 v19; // rax
  char v20; // [rsp+0h] [rbp-40h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned __int16 *)(a2 + 68);
  if ( (unsigned int)(v6 - 22) > 1 )
  {
    if ( !(_BYTE)qword_5025DA8 )
      return 0;
    v12 = qword_5025E88 | ((unsigned __int16)(v6 - 14) <= 4u);
    if ( v12 )
      return 0;
    v13 = *(_QWORD *)(a2 + 32);
    v14 = v13 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
    if ( v13 == v14 )
      return 0;
    v15 = *(_QWORD *)(a2 + 32);
    do
    {
      if ( *(_BYTE *)v15 == 5 )
      {
        v16 = *(_DWORD *)(v15 + 24);
        if ( v16 >= 0 )
        {
          v17 = (unsigned int)v16 >> 6;
          if ( (*(_QWORD *)(*(_QWORD *)(a1 + 1272) + 8 * v17) & (1LL << v16)) != 0 )
          {
            v18 = qword_5025DA8;
            if ( (_BYTE)qword_5025DA8 )
            {
              if ( !(_BYTE)qword_5025E88 && (*(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * v17) & (1LL << v16)) == 0 )
              {
                v19 = *(unsigned int *)(a3 + 8);
                if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
                {
                  v20 = qword_5025DA8;
                  v21 = v14;
                  sub_C8D5F0(a3, (const void *)(a3 + 16), v19 + 1, 4u, v14, 1);
                  v19 = *(unsigned int *)(a3 + 8);
                  v18 = v20;
                  v14 = v21;
                }
                *(_DWORD *)(*(_QWORD *)a3 + 4 * v19) = v16;
                v12 = v18;
                ++*(_DWORD *)(a3 + 8);
              }
            }
          }
        }
      }
      v15 += 40;
    }
    while ( v14 != v15 );
    if ( !v12 )
      return 0;
LABEL_26:
    *a4 = 1;
    return 1;
  }
  v7 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
  if ( v7 < 0 )
    return 0;
  v8 = 1LL << v7;
  v9 = (unsigned int)v7 >> 6;
  if ( (*(_QWORD *)(*(_QWORD *)(a1 + 1272) + 8 * v9) & (1LL << v7)) == 0 )
    return 0;
  v10 = *(unsigned int *)(a3 + 8);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v10 + 1, 4u, v9, v10 + 1);
    v10 = *(unsigned int *)(a3 + 8);
    v9 = (unsigned int)v7 >> 6;
    v8 = 1LL << v7;
  }
  *(_DWORD *)(*(_QWORD *)a3 + 4 * v10) = v7;
  ++*(_DWORD *)(a3 + 8);
  if ( *(_WORD *)(a2 + 68) != 23 )
  {
    if ( (_BYTE)qword_5025DA8 && !(_BYTE)qword_5025E88 && (*(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * v9) & v8) == 0 )
      return 0;
    goto LABEL_26;
  }
  *a4 = 0;
  return 1;
}
