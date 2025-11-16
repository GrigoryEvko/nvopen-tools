// Function: sub_2DB4530
// Address: 0x2db4530
//
__int64 __fastcall sub_2DB4530(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned int v4; // r12d
  char v5; // al
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 *v12; // rax
  int v13; // eax
  __int64 v14; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned int v19; // ecx

  v2 = *(_QWORD *)(a2 + 32);
  v3 = v2 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v3 != v2 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v2 == 12 )
        return 0;
      if ( !*(_BYTE *)v2 )
      {
        v4 = *(_DWORD *)(v2 + 8);
        if ( (*(_BYTE *)(v2 + 3) & 0x10) != 0
          && v4 - 1 <= 0x3FFFFFFE
          && !(unsigned __int8)sub_2EBF3A0(*(_QWORD *)(a1 + 16), v4) )
        {
          v16 = *(_QWORD *)(a1 + 8);
          v17 = *(_QWORD *)(v16 + 8);
          v18 = *(_QWORD *)(v16 + 56) + 2LL * (*(_DWORD *)(v17 + 24LL * v4 + 16) >> 12);
          v19 = *(_DWORD *)(v17 + 24LL * v4 + 16) & 0xFFF;
          if ( v18 )
          {
            do
            {
              v18 += 2;
              *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * (v19 >> 6)) |= 1LL << v19;
              v19 += *(__int16 *)(v18 - 2);
            }
            while ( *(_WORD *)(v18 - 2) );
          }
        }
        v5 = *(_BYTE *)(v2 + 4);
        if ( (v5 & 1) == 0
          && (v5 & 2) == 0
          && ((*(_BYTE *)(v2 + 3) & 0x10) == 0 || (*(_DWORD *)v2 & 0xFFF00) != 0)
          && (v4 & 0x80000000) != 0 )
        {
          v6 = sub_2EBEE10(*(_QWORD *)(a1 + 16), v4);
          v11 = v6;
          if ( v6 )
          {
            if ( *(_QWORD *)(a1 + 24) == *(_QWORD *)(v6 + 24) )
              break;
          }
        }
      }
LABEL_3:
      v2 += 40;
      if ( v3 == v2 )
        return 1;
    }
    if ( *(_BYTE *)(a1 + 532) )
    {
      v12 = *(__int64 **)(a1 + 512);
      v8 = *(unsigned int *)(a1 + 524);
      v7 = &v12[v8];
      if ( v12 != v7 )
      {
        while ( v11 != *v12 )
        {
          if ( v7 == ++v12 )
            goto LABEL_32;
        }
LABEL_20:
        v13 = *(_DWORD *)(v11 + 44);
        if ( (v13 & 4) == 0 && (v13 & 8) != 0 )
          LOBYTE(v14) = sub_2E88A90(v11, 512, 1);
        else
          v14 = (*(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL) >> 9) & 1LL;
        if ( (_BYTE)v14 )
          return 0;
        goto LABEL_3;
      }
LABEL_32:
      if ( (unsigned int)v8 < *(_DWORD *)(a1 + 520) )
      {
        *(_DWORD *)(a1 + 524) = v8 + 1;
        *v7 = v11;
        ++*(_QWORD *)(a1 + 504);
        goto LABEL_20;
      }
    }
    sub_C8CC70(a1 + 504, v11, (__int64)v7, v8, v9, v10);
    goto LABEL_20;
  }
  return 1;
}
