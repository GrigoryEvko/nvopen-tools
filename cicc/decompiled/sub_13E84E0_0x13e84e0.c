// Function: sub_13E84E0
// Address: 0x13e84e0
//
__int64 __fastcall sub_13E84E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v5; // edx
  __int64 v6; // rsi
  int v7; // edi
  __int64 *v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  _QWORD *i; // rbx
  _QWORD *v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rdx
  int v22; // eax
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdi
  __int64 v27; // rdi

  if ( *(_DWORD *)(a1 + 80) )
  {
    v15 = *(_QWORD **)(a1 + 72);
    v16 = &v15[10 * *(unsigned int *)(a1 + 88)];
    if ( v15 != v16 )
    {
      while ( 1 )
      {
        v17 = v15;
        if ( *v15 != -16 && *v15 != -8 )
          break;
        v15 += 10;
        if ( v16 == v15 )
          goto LABEL_2;
      }
      if ( v16 != v15 )
      {
        while ( 1 )
        {
          for ( i = v17 + 10; v16 != i; i += 10 )
          {
            if ( *i != -8 && *i != -16 )
              break;
          }
          v19 = (_QWORD *)v17[2];
          if ( (_QWORD *)v17[3] == v19 )
          {
            v21 = &v19[*((unsigned int *)v17 + 9)];
            if ( v19 == v21 )
            {
LABEL_47:
              v19 = v21;
            }
            else
            {
              while ( *v19 != a2 )
              {
                if ( v21 == ++v19 )
                  goto LABEL_47;
              }
            }
          }
          else
          {
            v19 = (_QWORD *)sub_16CC9F0(v17 + 1, a2);
            if ( *v19 == a2 )
            {
              v24 = v17[3];
              if ( v24 == v17[2] )
                v25 = *((unsigned int *)v17 + 9);
              else
                v25 = *((unsigned int *)v17 + 8);
              v21 = (_QWORD *)(v24 + 8 * v25);
            }
            else
            {
              v20 = v17[3];
              if ( v20 != v17[2] )
                goto LABEL_32;
              v19 = (_QWORD *)(v20 + 8LL * *((unsigned int *)v17 + 9));
              v21 = v19;
            }
          }
          if ( v21 != v19 )
          {
            *v19 = -2;
            v22 = *((_DWORD *)v17 + 10) + 1;
            *((_DWORD *)v17 + 10) = v22;
            if ( *((_DWORD *)v17 + 9) != v22 )
              goto LABEL_33;
            goto LABEL_41;
          }
LABEL_32:
          if ( *((_DWORD *)v17 + 9) != *((_DWORD *)v17 + 10) )
            goto LABEL_33;
LABEL_41:
          v23 = v17[3];
          if ( v23 != v17[2] )
            _libc_free(v23);
          *v17 = -16;
          --*(_DWORD *)(a1 + 80);
          ++*(_DWORD *)(a1 + 84);
LABEL_33:
          if ( v16 == i )
            break;
          v17 = i;
        }
      }
    }
  }
LABEL_2:
  result = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)result )
  {
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 40);
    v7 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16 * result);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_4:
      v10 = v8[1];
      if ( !v10 )
      {
LABEL_17:
        *v8 = -16;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
        return result;
      }
      if ( (*(_BYTE *)(v10 + 48) & 1) != 0 )
      {
        v12 = v10 + 56;
        v13 = v10 + 248;
      }
      else
      {
        v11 = *(unsigned int *)(v10 + 64);
        v12 = *(_QWORD *)(v10 + 56);
        if ( !(_DWORD)v11 )
          goto LABEL_50;
        v13 = v12 + 48 * v11;
      }
      do
      {
        if ( *(_QWORD *)v12 != -16 && *(_QWORD *)v12 != -8 && *(_DWORD *)(v12 + 8) == 3 )
        {
          if ( *(_DWORD *)(v12 + 40) > 0x40u )
          {
            v26 = *(_QWORD *)(v12 + 32);
            if ( v26 )
              j_j___libc_free_0_0(v26);
          }
          if ( *(_DWORD *)(v12 + 24) > 0x40u )
          {
            v27 = *(_QWORD *)(v12 + 16);
            if ( v27 )
              j_j___libc_free_0_0(v27);
          }
        }
        v12 += 48;
      }
      while ( v12 != v13 );
      if ( (*(_BYTE *)(v10 + 48) & 1) != 0 )
        goto LABEL_13;
      v12 = *(_QWORD *)(v10 + 56);
LABEL_50:
      j___libc_free_0(v12);
LABEL_13:
      *(_QWORD *)v10 = &unk_49EE2B0;
      v14 = *(_QWORD *)(v10 + 24);
      if ( v14 != -8 && v14 != 0 && v14 != -16 )
        sub_1649B30(v10 + 8);
      result = j_j___libc_free_0(v10, 248);
      goto LABEL_17;
    }
    while ( v9 != -8 )
    {
      result = v5 & (unsigned int)(v7 + result);
      v8 = (__int64 *)(v6 + 16LL * (unsigned int)result);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_4;
      ++v7;
    }
  }
  return result;
}
