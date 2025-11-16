// Function: sub_2DF7A30
// Address: 0x2df7a30
//
__int64 __fastcall sub_2DF7A30(__int64 a1, __int64 a2)
{
  __int64 v4; // r8
  __int64 v5; // rsi
  unsigned int v6; // r8d
  __int64 v7; // rcx
  unsigned __int64 v8; // rdi
  __int64 v9; // r11
  __int64 v10; // r9
  unsigned int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r9
  __int64 v16; // rsi
  unsigned __int64 v17; // r8
  __int64 v18; // rdx
  __int64 i; // rax
  unsigned __int64 v20; // rcx
  __int64 result; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  unsigned int v24; // edx
  unsigned __int64 v25; // r14
  __int64 v27; // rcx
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned int v30; // edi
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax

  v4 = a2 >> 1;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = v4 & 3;
  v7 = *(unsigned int *)(a1 + 16);
  v8 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = v5 + 16 * v7 - 16;
  v10 = *(_QWORD *)v9;
  v11 = v6 | *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v12 = *(_QWORD *)(*(_QWORD *)v9 + 16LL * (unsigned int)(*(_DWORD *)(v9 + 8) - 1) + 8);
  if ( (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12 >> 1) & 3) > v11 )
  {
    v27 = *(unsigned int *)(v9 + 12);
    for ( result = *(_DWORD *)((*(_QWORD *)(v10 + 16 * v27 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(*(__int64 *)(v10 + 16 * v27 + 8) >> 1) & 3;
          v11 >= (unsigned int)result;
          result = *(_DWORD *)((*(_QWORD *)(v10 + 16 * v27 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(*(__int64 *)(v10 + 16 * v27 + 8) >> 1) & 3 )
    {
      v27 = (unsigned int)(v27 + 1);
    }
    *(_DWORD *)(v9 + 12) = v27;
  }
  else
  {
    *(_DWORD *)(a1 + 16) = v7 - 1;
    if ( (_DWORD)v7 == 2 )
    {
      v22 = *(unsigned int *)(v5 + 12);
      v23 = *(_QWORD *)a1;
      v24 = *(_DWORD *)(v5 + 12);
    }
    else
    {
      v13 = (unsigned int)(v7 - 3);
      v14 = v5 + 16 * v13;
      if ( (_DWORD)v13 )
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(*(_QWORD *)v14 + 8LL * *(unsigned int *)(v14 + 12) + 96);
          if ( (*(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v15 >> 1) & 3) > (v6
                                                                                                | *(_DWORD *)(v8 + 24)) )
            break;
          --*(_DWORD *)(a1 + 16);
          v14 -= 16;
          LODWORD(v13) = v13 - 1;
          if ( !(_DWORD)v13 )
            goto LABEL_28;
        }
        v16 = 16LL * (unsigned int)(v13 + 1) + v5;
        v17 = *(_QWORD *)v16;
        v18 = *(unsigned int *)(v16 + 12);
        for ( i = *(_QWORD *)(*(_QWORD *)v16 + 8 * v18 + 96); ; i = *(_QWORD *)(v17 + 8 * v18 + 96) )
        {
          v20 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(i >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                            | (unsigned int)(a2 >> 1)
                                                                                            & 3) )
            break;
          v18 = (unsigned int)(v18 + 1);
        }
        *(_DWORD *)(v16 + 12) = v18;
        return sub_2DF5F90(a1, a2, v18, v20, v17);
      }
LABEL_28:
      v22 = *(unsigned int *)(v5 + 12);
      v23 = *(_QWORD *)a1;
      v24 = *(_DWORD *)(v5 + 12);
      v28 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v22 + 80);
      v29 = v28 >> 1;
      v10 = v28 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_DWORD *)(v10 + 24) | (unsigned int)(v29 & 3)) > (*(_DWORD *)(v8 + 24) | v6) )
      {
        v17 = *(_QWORD *)(v5 + 16);
        v18 = *(unsigned int *)(v5 + 28);
        v30 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
        v31 = *(_QWORD *)(v17 + 8 * v18 + 96);
        v20 = v31 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v30 >= (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v31 >> 1) & 3) )
        {
          do
          {
            v18 = (unsigned int)(v18 + 1);
            v32 = *(_QWORD *)(v17 + 8 * v18 + 96);
            v33 = v32 >> 1;
            v20 = v32 & 0xFFFFFFFFFFFFFFF8LL;
          }
          while ( (*(_DWORD *)(v20 + 24) | (unsigned int)(v33 & 3)) <= v30 );
        }
        *(_DWORD *)(v5 + 28) = v18;
        return sub_2DF5F90(a1, a2, v18, v20, v17);
      }
    }
    v17 = *(unsigned int *)(v23 + 164);
    if ( (_DWORD)v17 != v24 )
    {
      v10 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3;
      while ( (*(_DWORD *)((*(_QWORD *)(v23 + 8 + 8 * v22 + 72) & 0xFFFFFFFFFFFFFFF8LL) + 24)
             | (unsigned int)(*(__int64 *)(v23 + 8 + 8 * v22 + 72) >> 1) & 3) <= (unsigned int)v10 )
      {
        if ( (_DWORD)v17 == ++v24 )
        {
          v22 = (unsigned int)v17;
          break;
        }
        v22 = v24;
      }
    }
    v18 = *(unsigned int *)(v23 + 160);
    result = *(unsigned int *)(a1 + 20);
    *(_DWORD *)(a1 + 16) = 0;
    if ( (_DWORD)v18 )
      v23 += 8;
    v20 = v22 << 32;
    v25 = v20 | (unsigned int)v17;
    if ( !(_DWORD)result )
    {
      result = sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), 1u, 0x10u, v17, v10);
      v5 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16);
    }
    *(_QWORD *)v5 = v23;
    *(_QWORD *)(v5 + 8) = v25;
    if ( (*(_DWORD *)(a1 + 16))++ != -1 )
    {
      result = *(_QWORD *)(a1 + 8);
      if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
        return sub_2DF5F90(a1, a2, v18, v20, v17);
    }
  }
  return result;
}
