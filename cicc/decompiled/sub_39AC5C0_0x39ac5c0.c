// Function: sub_39AC5C0
// Address: 0x39ac5c0
//
__int64 __fastcall sub_39AC5C0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int16 v5; // ax
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 *v11; // rsi
  __int64 v12; // r9
  int v13; // eax
  __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // esi
  int v19; // r11d
  __int64 v20; // rdx

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 == *(_QWORD *)(a1 + 24) )
    goto LABEL_44;
  v3 = *(_QWORD *)(a1 + 32);
  while ( 2 )
  {
    v4 = v2 + 24;
    if ( v2 + 24 == v3 )
      goto LABEL_26;
    do
    {
      if ( !*(_BYTE *)(a1 + 64) && *(_DWORD *)(a1 + 56) != *(_DWORD *)(a1 + 68) )
      {
        v5 = *(_WORD *)(v3 + 46);
        if ( (v5 & 4) == 0 && (v5 & 8) != 0 )
        {
          if ( !sub_1E15D00(v3, 0x10u, 1) )
            goto LABEL_9;
        }
        else if ( (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) & 0x10LL) == 0 )
        {
          goto LABEL_9;
        }
        if ( !(unsigned __int8)sub_39AA510(v3) )
        {
          v17 = *(_QWORD *)(a1 + 8);
          *(_QWORD *)(a1 + 48) = 0;
          *(_QWORD *)(a1 + 8) = 0;
          *(_QWORD *)(a1 + 40) = v17;
          *(_DWORD *)(a1 + 56) = *(_DWORD *)(a1 + 68);
          v14 = *(_QWORD *)(a1 + 32);
          if ( !v14 )
            BUG();
          if ( (*(_BYTE *)v14 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v14 + 46) & 8) != 0 )
              v14 = *(_QWORD *)(v14 + 8);
          }
          goto LABEL_17;
        }
      }
LABEL_9:
      if ( **(_WORD **)(v3 + 16) == 3 )
      {
        v6 = *(_QWORD *)(a1 + 8);
        v7 = *(_QWORD *)(*(_QWORD *)(v3 + 32) + 24LL);
        if ( v7 == v6 )
        {
          *(_BYTE *)(a1 + 64) = 0;
        }
        else
        {
          v8 = *(unsigned int *)(*(_QWORD *)a1 + 120LL);
          if ( (_DWORD)v8 )
          {
            v9 = *(_QWORD *)(*(_QWORD *)a1 + 104LL);
            v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v11 = (__int64 *)(v9 + 24LL * v10);
            v12 = *v11;
            if ( v7 == *v11 )
            {
LABEL_13:
              if ( v11 != (__int64 *)(v9 + 24 * v8) )
              {
                v13 = *((_DWORD *)v11 + 2);
                *(_BYTE *)(a1 + 64) = 1;
                if ( v13 != *(_DWORD *)(a1 + 56) )
                {
                  *(_QWORD *)(a1 + 40) = v6;
                  *(_QWORD *)(a1 + 48) = v7;
                  *(_DWORD *)(a1 + 56) = v13;
                  *(_QWORD *)(a1 + 8) = v11[2];
                  v14 = *(_QWORD *)(a1 + 32);
                  if ( !v14 )
                    BUG();
                  if ( (*(_BYTE *)v14 & 4) == 0 )
                  {
                    while ( (*(_BYTE *)(v14 + 46) & 8) != 0 )
                      v14 = *(_QWORD *)(v14 + 8);
                  }
LABEL_17:
                  result = *(_QWORD *)(v14 + 8);
                  *(_QWORD *)(a1 + 32) = result;
                  return result;
                }
                *(_QWORD *)(a1 + 8) = v11[2];
              }
            }
            else
            {
              v18 = 1;
              while ( v12 != -8 )
              {
                v19 = v18 + 1;
                v10 = (v8 - 1) & (v18 + v10);
                v11 = (__int64 *)(v9 + 24LL * v10);
                v12 = *v11;
                if ( v7 == *v11 )
                  goto LABEL_13;
                v18 = v19;
              }
            }
          }
        }
      }
      v16 = *(_QWORD *)(a1 + 32);
      if ( !v16 )
        BUG();
      if ( (*(_BYTE *)v16 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v16 + 46) & 8) != 0 )
          v16 = *(_QWORD *)(v16 + 8);
      }
      v3 = *(_QWORD *)(v16 + 8);
      *(_QWORD *)(a1 + 32) = v3;
    }
    while ( v4 != v3 );
    v2 = *(_QWORD *)(a1 + 16);
LABEL_26:
    v2 = *(_QWORD *)(v2 + 8);
    *(_QWORD *)(a1 + 16) = v2;
    if ( v2 != *(_QWORD *)(a1 + 24) )
    {
      v3 = *(_QWORD *)(v2 + 32);
      *(_QWORD *)(a1 + 32) = v3;
      continue;
    }
    break;
  }
LABEL_44:
  result = *(unsigned int *)(a1 + 68);
  if ( *(_DWORD *)(a1 + 56) == (_DWORD)result )
  {
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 48) = 0;
    *(_DWORD *)(a1 + 56) = result;
    *(_QWORD *)(a1 + 40) = v20;
  }
  return result;
}
