// Function: sub_2C72C30
// Address: 0x2c72c30
//
__int64 __fastcall sub_2C72C30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v12; // r13
  __int64 v13; // r14
  _QWORD *v14; // rdx
  _QWORD *v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rax
  _BYTE *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdi
  __int64 *v32; // rax
  __int64 v33; // rax
  int v34; // eax
  unsigned int v35; // [rsp+Ch] [rbp-44h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v3 = **(_QWORD **)(a1 + 32);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4 )
LABEL_67:
    BUG();
  while ( 1 )
  {
    v8 = *(_QWORD *)(v4 + 24);
    v4 = *(_QWORD *)(v4 + 8);
    if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
      break;
    if ( !v4 )
      goto LABEL_67;
  }
  do
  {
    if ( !v4 )
      return 0;
    v9 = *(_QWORD *)(v4 + 24);
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( (unsigned __int8)(*(_BYTE *)v9 - 30) > 0xAu );
  if ( !v4 )
  {
LABEL_12:
    v12 = *(_QWORD *)(v8 + 40);
    v13 = *(_QWORD *)(v9 + 40);
    if ( *(_BYTE *)(a1 + 84) )
    {
      v14 = *(_QWORD **)(a1 + 64);
      v15 = &v14[*(unsigned int *)(a1 + 76)];
      if ( v14 == v15 )
        return 0;
      v16 = *(_QWORD **)(a1 + 64);
      do
      {
        if ( *v16 == v13 )
          goto LABEL_17;
        ++v16;
      }
      while ( v15 != v16 );
LABEL_49:
      while ( *v14 != v12 )
      {
        if ( v16 == ++v14 )
          return 0;
      }
      v29 = v12;
      v12 = v13;
      v13 = v29;
    }
    else
    {
      v36 = v3;
      v30 = sub_C8CA60(a1 + 56, *(_QWORD *)(v9 + 40));
      v31 = a1 + 56;
      v3 = v36;
      if ( v30 )
      {
        if ( *(_BYTE *)(a1 + 84) )
        {
          v14 = *(_QWORD **)(a1 + 64);
          v15 = &v14[*(unsigned int *)(a1 + 76)];
          if ( v15 != v14 )
          {
LABEL_17:
            while ( *v14 != v12 )
            {
              if ( ++v14 == v15 )
                goto LABEL_19;
            }
            return 0;
          }
        }
        else
        {
          v32 = sub_C8CA60(v31, v12);
          v3 = v36;
          if ( v32 )
            return 0;
        }
      }
      else
      {
        if ( *(_BYTE *)(a1 + 84) )
        {
          v14 = *(_QWORD **)(a1 + 64);
          v16 = &v14[*(unsigned int *)(a1 + 76)];
          if ( v16 != v14 )
            goto LABEL_49;
          return 0;
        }
        if ( !sub_C8CA60(v31, v12) )
          return 0;
        v33 = v12;
        v3 = v36;
        v12 = v13;
        v13 = v33;
      }
    }
LABEL_19:
    v17 = *(_QWORD *)(v3 + 56);
    if ( a2 )
      v17 = *(_QWORD *)(a2 + 32);
    while ( 1 )
    {
      if ( !v17 )
        BUG();
      if ( *(_BYTE *)(v17 - 24) != 84 )
        return 0;
      if ( !a3 || a3 == *(_QWORD *)(v17 - 16) )
      {
        v18 = *(_QWORD *)(v17 - 32);
        v19 = 0x1FFFFFFFE0LL;
        if ( (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) != 0 )
        {
          v20 = 0;
          do
          {
            if ( v12 == *(_QWORD *)(v18 + 32LL * *(unsigned int *)(v17 + 48) + 8 * v20) )
            {
              v19 = 32 * v20;
              goto LABEL_32;
            }
            ++v20;
          }
          while ( (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) != (_DWORD)v20 );
          v19 = 0x1FFFFFFFE0LL;
        }
LABEL_32:
        v21 = *(_BYTE **)(v18 + v19);
        if ( *v21 == 17 && sub_AC30F0((__int64)v21) )
        {
          v22 = *(_QWORD *)(v17 - 32);
          v23 = 0x1FFFFFFFE0LL;
          if ( (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) != 0 )
          {
            v24 = 0;
            do
            {
              if ( v13 == *(_QWORD *)(v22 + 32LL * *(unsigned int *)(v17 + 48) + 8 * v24) )
              {
                v23 = 32 * v24;
                goto LABEL_39;
              }
              ++v24;
            }
            while ( (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) != (_DWORD)v24 );
            v23 = 0x1FFFFFFFE0LL;
          }
LABEL_39:
          v25 = *(_QWORD *)(v22 + v23);
          if ( *(_BYTE *)v25 == 42 )
          {
            v26 = (*(_BYTE *)(v25 + 7) & 0x40) != 0
                ? *(_QWORD **)(v25 - 8)
                : (_QWORD *)(v25 - 32LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF));
            v10 = v17 - 24;
            if ( v17 - 24 == *v26 )
            {
              v27 = v26[4];
              if ( *(_BYTE *)v27 == 17 )
              {
                v35 = *(_DWORD *)(v27 + 32);
                if ( v35 > 0x40 )
                {
                  v37 = v26[4];
                  v34 = sub_C444A0(v27 + 24);
                  v10 = v17 - 24;
                  if ( v35 - v34 > 0x40 )
                    goto LABEL_22;
                  v28 = **(_QWORD **)(v37 + 24);
                }
                else
                {
                  v28 = *(_QWORD *)(v27 + 24);
                }
                if ( v28 == 1 )
                  return v10;
              }
            }
          }
        }
      }
LABEL_22:
      v17 = *(_QWORD *)(v17 + 8);
    }
  }
  while ( (unsigned __int8)(**(_BYTE **)(v4 + 24) - 30) > 0xAu )
  {
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      goto LABEL_12;
  }
  return 0;
}
