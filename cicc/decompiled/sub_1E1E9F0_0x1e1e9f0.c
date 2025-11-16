// Function: sub_1E1E9F0
// Address: 0x1e1e9f0
//
__int64 __fastcall sub_1E1E9F0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // rdi
  unsigned int i; // eax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r13
  int v9; // eax
  __int64 v10; // rbx
  __int64 v11; // r10
  _QWORD *v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rcx
  _QWORD *v18; // rcx
  __int64 v20; // rax
  _BOOL4 v21; // eax
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // r10
  __int64 v25; // rax
  __int64 v26; // [rsp+18h] [rbp-88h]
  __int64 v27; // [rsp+18h] [rbp-88h]
  _QWORD *v28; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-78h]
  unsigned int v30; // [rsp+2Ch] [rbp-74h]
  _QWORD v31[14]; // [rsp+30h] [rbp-70h] BYREF

  v4 = v31;
  v30 = 8;
  v31[0] = a2;
  v28 = v31;
  for ( i = 1; ; a2 = v4[i - 1] )
  {
    v6 = *(unsigned int *)(a2 + 40);
    v7 = *(_QWORD *)(a2 + 32);
    v29 = --i;
    v8 = v7 + 40 * v6;
    if ( v8 != v7 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)v7 )
          goto LABEL_4;
        if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
          goto LABEL_4;
        v9 = *(_DWORD *)(v7 + 8);
        if ( v9 >= 0 )
          goto LABEL_4;
        v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 264) + 24LL) + 16LL * (v9 & 0x7FFFFFFF) + 8);
        if ( !v10 )
          goto LABEL_4;
        if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
        {
          while ( 1 )
          {
            v10 = *(_QWORD *)(v10 + 32);
            if ( !v10 )
              break;
            if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
              goto LABEL_10;
          }
          v7 += 40;
          if ( v8 == v7 )
          {
LABEL_28:
            i = v29;
            v4 = v28;
            break;
          }
        }
        else
        {
LABEL_10:
          v11 = *(_QWORD *)(v10 + 16);
LABEL_11:
          LOBYTE(v2) = **(_WORD **)(v11 + 16) == 45 || **(_WORD **)(v11 + 16) == 0;
          if ( (_BYTE)v2 )
          {
            v26 = v11;
            if ( sub_1DA1810(*(_QWORD *)(a1 + 608) + 56LL, *(_QWORD *)(v11 + 24)) )
              goto LABEL_21;
            v12 = *(_QWORD **)(a1 + 624);
            v13 = 8LL * *(unsigned int *)(a1 + 632);
            v14 = &v12[(unsigned __int64)v13 / 8];
            v15 = *(_QWORD *)(v26 + 24);
            v16 = v13 >> 3;
            v17 = v13 >> 5;
            if ( !v17 )
              goto LABEL_32;
            v18 = &v12[4 * v17];
            do
            {
              if ( v15 == *v12 )
                goto LABEL_20;
              if ( v15 == v12[1] )
              {
                ++v12;
                goto LABEL_20;
              }
              if ( v15 == v12[2] )
              {
                v12 += 2;
                goto LABEL_20;
              }
              if ( v15 == v12[3] )
              {
                v12 += 3;
                goto LABEL_20;
              }
              v12 += 4;
            }
            while ( v18 != v12 );
            v16 = v14 - v12;
LABEL_32:
            if ( v16 != 2 )
            {
              if ( v16 != 3 )
              {
                if ( v16 != 1 )
                  goto LABEL_35;
LABEL_46:
                if ( v15 != *v12 )
                  goto LABEL_35;
LABEL_20:
                if ( v14 == v12 )
                {
LABEL_35:
                  v20 = *(_QWORD *)(v10 + 16);
                  goto LABEL_36;
                }
LABEL_21:
                v4 = v28;
                goto LABEL_22;
              }
              if ( v15 == *v12 )
                goto LABEL_20;
              ++v12;
            }
            if ( v15 == *v12 )
              goto LABEL_20;
            ++v12;
            goto LABEL_46;
          }
          if ( **(_WORD **)(v11 + 16) == 15 )
          {
            v27 = v11;
            v21 = sub_1DA1810(*(_QWORD *)(a1 + 608) + 56LL, *(_QWORD *)(v11 + 24));
            v24 = v27;
            if ( v21 )
            {
              v25 = v29;
              if ( v29 >= v30 )
              {
                sub_16CD150((__int64)&v28, v31, 0, 8, v22, v23);
                v25 = v29;
                v24 = v27;
              }
              v28[v25] = v24;
              ++v29;
            }
            goto LABEL_35;
          }
          v20 = v11;
LABEL_36:
          while ( 1 )
          {
            v10 = *(_QWORD *)(v10 + 32);
            if ( !v10 )
              break;
            if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
            {
              v11 = *(_QWORD *)(v10 + 16);
              if ( v11 != v20 )
                goto LABEL_11;
            }
          }
LABEL_4:
          v7 += 40;
          if ( v8 == v7 )
            goto LABEL_28;
        }
      }
    }
    if ( !i )
      break;
  }
  v2 = 0;
LABEL_22:
  if ( v4 != v31 )
    _libc_free((unsigned __int64)v4);
  return v2;
}
