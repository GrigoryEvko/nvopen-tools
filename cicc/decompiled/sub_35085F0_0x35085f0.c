// Function: sub_35085F0
// Address: 0x35085f0
//
char __fastcall sub_35085F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 i; // r13
  unsigned int v19; // esi

  v8 = *(__int64 **)(a2 + 112);
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  while ( v9 != v8 )
  {
    v10 = *v8++;
    sub_35081E0(a1, v10);
  }
  v11 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v12 = v11;
  if ( v11 == a2 + 48 )
    return v11;
  if ( !v11 )
    BUG();
  v13 = *(_QWORD *)v11;
  v14 = *(_DWORD *)(v12 + 44);
  v15 = v14 & 4;
  v16 = v14 & 0xFFFFFF;
  if ( (v13 & 4) != 0 )
  {
    if ( (_DWORD)v15 )
      goto LABEL_7;
  }
  else if ( (_DWORD)v15 )
  {
    while ( 1 )
    {
      v12 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      LOBYTE(v16) = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 44);
      if ( (v16 & 4) == 0 )
        break;
      v13 = *(_QWORD *)v12;
    }
  }
  v16 &= 8u;
  if ( (_DWORD)v16 )
  {
    LOBYTE(v11) = sub_2E88A90(v12, 32, 1);
    goto LABEL_8;
  }
LABEL_7:
  v11 = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 5) & 1LL;
LABEL_8:
  if ( (_BYTE)v11 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL);
    if ( *(_BYTE *)(v11 + 120) )
    {
      v17 = *(_QWORD *)(v11 + 96);
      for ( i = *(_QWORD *)(v11 + 104); i != v17; LOBYTE(v11) = sub_3507B80(a1, v19, v16, v15, a5, a6) )
      {
        while ( !*(_BYTE *)(v17 + 8) )
        {
          v17 += 12;
          if ( i == v17 )
            return v11;
        }
        v19 = *(_DWORD *)v17;
        v17 += 12;
      }
    }
  }
  return v11;
}
