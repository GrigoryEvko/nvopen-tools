// Function: sub_1411E70
// Address: 0x1411e70
//
void __fastcall sub_1411E70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // r9d
  unsigned int v8; // edx
  __int64 *v9; // rbx
  __int64 v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // eax
  unsigned __int64 v17; // rdi

  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v5 )
  {
    v7 = 1;
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v6 + 80LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      v11 = (_QWORD *)v9[2];
      if ( (_QWORD *)v9[3] != v11 )
        goto LABEL_4;
LABEL_12:
      v13 = &v11[*((unsigned int *)v9 + 9)];
      if ( v11 == v13 )
      {
LABEL_15:
        v11 = v13;
      }
      else
      {
        while ( a3 != *v11 )
        {
          if ( v13 == ++v11 )
            goto LABEL_15;
        }
      }
      goto LABEL_19;
    }
    while ( v10 != -8 )
    {
      v8 = (v5 - 1) & (v7 + v8);
      v9 = (__int64 *)(v6 + 80LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      ++v7;
    }
  }
  v9 = (__int64 *)(v6 + 80 * v5);
  v11 = (_QWORD *)v9[2];
  if ( (_QWORD *)v9[3] == v11 )
    goto LABEL_12;
LABEL_4:
  v11 = (_QWORD *)sub_16CC9F0(v9 + 1, a3);
  if ( a3 == *v11 )
  {
    v14 = v9[3];
    if ( v14 == v9[2] )
      v15 = *((unsigned int *)v9 + 9);
    else
      v15 = *((unsigned int *)v9 + 8);
    v13 = (_QWORD *)(v14 + 8 * v15);
  }
  else
  {
    v12 = v9[3];
    if ( v12 != v9[2] )
      goto LABEL_6;
    v11 = (_QWORD *)(v12 + 8LL * *((unsigned int *)v9 + 9));
    v13 = v11;
  }
LABEL_19:
  if ( v13 != v11 )
  {
    *v11 = -2;
    v16 = *((_DWORD *)v9 + 10) + 1;
    *((_DWORD *)v9 + 10) = v16;
    if ( *((_DWORD *)v9 + 9) != v16 )
      return;
    goto LABEL_21;
  }
LABEL_6:
  if ( *((_DWORD *)v9 + 9) != *((_DWORD *)v9 + 10) )
    return;
LABEL_21:
  v17 = v9[3];
  if ( v17 != v9[2] )
    _libc_free(v17);
  *v9 = -16;
  --*(_DWORD *)(a1 + 16);
  ++*(_DWORD *)(a1 + 20);
}
