// Function: sub_1B904E0
// Address: 0x1b904e0
//
__int64 __fastcall sub_1B904E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  unsigned int v10; // ebx
  __int64 v11; // r15
  __int64 v12; // rdx
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  int v17; // r9d
  __int64 v18; // rdi
  unsigned int v19; // ecx
  unsigned int v20; // esi
  int *v21; // r11
  int v22; // r8d
  __int64 v23; // r13
  __int64 v24; // rdx
  int v25; // r10d
  unsigned int v26; // r15d
  int v27; // r11d
  int v28; // r11d
  int v29; // ecx

  v5 = sub_1263B40(a2, " +\n");
  sub_16E2CE0(a3, v5);
  v6 = sub_1263B40(v5, "\"INTERLEAVE-GROUP with factor ");
  v7 = sub_16E7A90(v6, **(unsigned int **)(a1 + 40));
  sub_1263B40(v7, " at ");
  sub_15537D0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL), a2, 0, 0);
  result = sub_1263B40(a2, "\\l\"");
  v9 = *(_QWORD *)(a1 + 40);
  if ( *(_DWORD *)v9 )
  {
    v10 = 0;
    do
    {
      while ( 1 )
      {
        result = *(unsigned int *)(v9 + 40);
        if ( (_DWORD)result )
        {
          v17 = result - 1;
          v18 = *(_QWORD *)(v9 + 24);
          v19 = v10 + *(_DWORD *)(v9 + 48);
          v20 = (result - 1) & (37 * v19);
          v21 = (int *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v19 != *v21 )
          {
            v25 = *v21;
            v26 = (result - 1) & (37 * v19);
            v27 = 1;
            while ( v25 != 0x7FFFFFFF )
            {
              v26 = v17 & (v27 + v26);
              v25 = *(_DWORD *)(v18 + 16LL * v26);
              if ( v19 == v25 )
              {
                v28 = 1;
                while ( v22 != 0x7FFFFFFF )
                {
                  v29 = v28 + 1;
                  v20 = v17 & (v28 + v20);
                  v21 = (int *)(v18 + 16LL * v20);
                  v22 = *v21;
                  if ( v25 == *v21 )
                    goto LABEL_13;
                  v28 = v29;
                }
                result *= 16;
                v21 = (int *)(v18 + result);
                goto LABEL_13;
              }
              ++v27;
            }
            goto LABEL_10;
          }
LABEL_13:
          v23 = *((_QWORD *)v21 + 1);
          if ( v23 )
            break;
        }
LABEL_10:
        if ( ++v10 >= *(_DWORD *)v9 )
          return result;
      }
      v24 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v24) > 2 )
      {
        *(_BYTE *)(v24 + 2) = 10;
        v11 = a2;
        *(_WORD *)v24 = 11040;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      else
      {
        v11 = sub_16E7EE0(a2, " +\n", 3u);
      }
      sub_16E2CE0(a3, v11);
      v12 = *(_QWORD *)(v11 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v11 + 16) - v12) <= 2 )
      {
        v11 = sub_16E7EE0(v11, "\"  ", 3u);
      }
      else
      {
        *(_BYTE *)(v12 + 2) = 32;
        *(_WORD *)v12 = 8226;
        *(_QWORD *)(v11 + 24) += 3LL;
      }
      sub_1BE27E0(v11, v23);
      v13 = *(_BYTE **)(v11 + 24);
      if ( *(_BYTE **)(v11 + 16) == v13 )
      {
        v11 = sub_16E7EE0(v11, " ", 1u);
      }
      else
      {
        *v13 = 32;
        ++*(_QWORD *)(v11 + 24);
      }
      v14 = sub_16E7A90(v11, v10);
      v15 = *(_QWORD *)(v14 + 24);
      v16 = v14;
      if ( (unsigned __int64)(*(_QWORD *)(v14 + 16) - v15) > 2 )
      {
        *(_BYTE *)(v15 + 2) = 34;
        *(_WORD *)v15 = 27740;
        result = a1;
        *(_QWORD *)(v16 + 24) += 3LL;
        v9 = *(_QWORD *)(a1 + 40);
        goto LABEL_10;
      }
      ++v10;
      sub_16E7EE0(v14, "\\l\"", 3u);
      result = a1;
      v9 = *(_QWORD *)(a1 + 40);
    }
    while ( v10 < *(_DWORD *)v9 );
  }
  return result;
}
