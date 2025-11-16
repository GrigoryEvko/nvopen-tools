// Function: sub_2ED19B0
// Address: 0x2ed19b0
//
__int64 __fastcall sub_2ED19B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 i; // r15
  __int64 (*v9)(); // rax
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 j; // r14
  __int64 (*v14)(); // rax
  char v15; // al
  unsigned int v16; // r10d
  unsigned int v17; // eax
  __int64 v18; // rax
  char v19; // al
  unsigned __int8 v22; // [rsp+1Bh] [rbp-35h]
  unsigned int v23; // [rsp+1Ch] [rbp-34h]
  unsigned int v24; // [rsp+1Ch] [rbp-34h]
  unsigned int v25; // [rsp+1Ch] [rbp-34h]

  for ( i = sub_2E311E0(a1); a2 != i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v9 = *(__int64 (**)())(*(_QWORD *)a5 + 1336LL);
      if ( v9 != sub_2E2F9B0 )
      {
        v22 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v9)(a5, i, 0);
        if ( v22 )
        {
          v11 = *(_QWORD *)(a3 + 32);
          v12 = v11 + 40LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF);
          if ( v12 != v11 )
          {
            for ( j = *(_QWORD *)(a3 + 32); v12 != j; j += 40 )
            {
              if ( !*(_BYTE *)j )
              {
                v16 = *(_DWORD *)(j + 8);
                if ( v16 )
                {
                  if ( (*(_BYTE *)(j + 3) & 0x10) != 0 )
                  {
                    v24 = *(_DWORD *)(j + 8);
                    if ( (unsigned int)sub_2E89C70(i, v16, a4, 0) != -1 )
                      return v22;
                    v17 = sub_2E8E710(i, v24, a4, 0, 1);
                    if ( v17 != -1 )
                    {
                      v18 = *(_QWORD *)(i + 32) + 40LL * v17;
                      if ( v18 )
                      {
                        if ( (((*(_BYTE *)(v18 + 3) & 0x10) != 0) & (*(_BYTE *)(v18 + 3) >> 6)) == 0 )
                          return v22;
                      }
                    }
                  }
                  else if ( v16 - 1 > 0x3FFFFFFE
                         || ((v14 = *(__int64 (**)())(*(_QWORD *)a5 + 32LL), v14 == sub_2E4EE60)
                          || (v25 = *(_DWORD *)(j + 8),
                              v19 = ((__int64 (__fastcall *)(__int64, __int64))v14)(a5, j),
                              v16 = v25,
                              !v19))
                         && (!a6 || (v23 = v16, v15 = sub_2EBF3A0(a6, v16), v16 = v23, !v15)) )
                  {
                    if ( (unsigned int)sub_2E8E710(i, v16, a4, 0, 1) != -1 )
                      return v22;
                  }
                }
              }
            }
          }
        }
      }
      if ( !i )
        BUG();
      if ( (*(_BYTE *)i & 4) == 0 )
        break;
      i = *(_QWORD *)(i + 8);
      if ( a2 == i )
        return 0;
    }
    while ( (*(_BYTE *)(i + 44) & 8) != 0 )
      i = *(_QWORD *)(i + 8);
  }
  return 0;
}
