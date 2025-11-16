// Function: sub_1C27910
// Address: 0x1c27910
//
void __fastcall sub_1C27910(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // r13d
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // ecx
  int *v9; // rax
  int v10; // edx
  int v11; // eax
  int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // edx
  int v15; // ecx
  int v16; // edi
  int v17; // r11d
  int *v18; // r10
  int v19; // ecx
  int v20; // eax
  int v21; // edx
  __int64 v22; // rdi
  int *v23; // r8
  unsigned int v24; // r14d
  int v25; // r9d
  int v26; // esi
  int v27; // r10d
  int *v28; // r9

  if ( a2 )
  {
    v4 = a1 + 1048;
    v5 = *(_DWORD *)(a2 + 8);
    v6 = *(_DWORD *)(a1 + 1072);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 1056);
      v8 = (v6 - 1) & (37 * v5);
      v9 = (int *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v5 == *v9 )
      {
LABEL_4:
        *((_QWORD *)v9 + 1) = a2;
        return;
      }
      v17 = 1;
      v18 = 0;
      while ( v10 != 0x7FFFFFFF )
      {
        if ( !v18 && v10 == 0x80000000 )
          v18 = v9;
        v8 = (v6 - 1) & (v17 + v8);
        v9 = (int *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v5 == *v9 )
          goto LABEL_4;
        ++v17;
      }
      v19 = *(_DWORD *)(a1 + 1064);
      if ( v18 )
        v9 = v18;
      ++*(_QWORD *)(a1 + 1048);
      v15 = v19 + 1;
      if ( 4 * v15 < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 1068) - v15 > v6 >> 3 )
        {
LABEL_9:
          *(_DWORD *)(a1 + 1064) = v15;
          if ( *v9 != 0x7FFFFFFF )
            --*(_DWORD *)(a1 + 1068);
          *v9 = v5;
          *((_QWORD *)v9 + 1) = 0;
          goto LABEL_4;
        }
        sub_12EABE0(v4, v6);
        v20 = *(_DWORD *)(a1 + 1072);
        if ( v20 )
        {
          v21 = v20 - 1;
          v22 = *(_QWORD *)(a1 + 1056);
          v23 = 0;
          v24 = (v20 - 1) & (37 * v5);
          v25 = 1;
          v15 = *(_DWORD *)(a1 + 1064) + 1;
          v9 = (int *)(v22 + 16LL * v24);
          v26 = *v9;
          if ( v5 != *v9 )
          {
            while ( v26 != 0x7FFFFFFF )
            {
              if ( !v23 && v26 == 0x80000000 )
                v23 = v9;
              v24 = v21 & (v25 + v24);
              v9 = (int *)(v22 + 16LL * v24);
              v26 = *v9;
              if ( v5 == *v9 )
                goto LABEL_9;
              ++v25;
            }
            if ( v23 )
              v9 = v23;
          }
          goto LABEL_9;
        }
LABEL_44:
        ++*(_DWORD *)(a1 + 1064);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 1048);
    }
    sub_12EABE0(v4, 2 * v6);
    v11 = *(_DWORD *)(a1 + 1072);
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = *(_QWORD *)(a1 + 1056);
      v14 = (v11 - 1) & (37 * v5);
      v15 = *(_DWORD *)(a1 + 1064) + 1;
      v9 = (int *)(v13 + 16LL * v14);
      v16 = *v9;
      if ( v5 != *v9 )
      {
        v27 = 1;
        v28 = 0;
        while ( v16 != 0x7FFFFFFF )
        {
          if ( !v28 && v16 == 0x80000000 )
            v28 = v9;
          v14 = v12 & (v27 + v14);
          v9 = (int *)(v13 + 16LL * v14);
          v16 = *v9;
          if ( v5 == *v9 )
            goto LABEL_9;
          ++v27;
        }
        if ( v28 )
          v9 = v28;
      }
      goto LABEL_9;
    }
    goto LABEL_44;
  }
}
