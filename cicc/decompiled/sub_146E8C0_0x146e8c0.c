// Function: sub_146E8C0
// Address: 0x146e8c0
//
__int64 __fastcall sub_146E8C0(__int64 a1, __int64 *a2, unsigned __int8 a3, __int64 a4, char a5)
{
  unsigned __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // r14
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // r13
  __int64 v18; // rsi
  __int64 *v19; // r15
  __int64 v20; // rax
  _QWORD *v21; // r13
  _QWORD *v22; // r15
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 *v25; // rax
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  *(_BYTE *)(a1 + 48) = a5;
  *(_QWORD *)(a1 + 40) = (4LL * a3) | a4 & 0xFFFFFFFFFFFFFFFBLL;
  v7 = *((unsigned int *)a2 + 2);
  if ( (unsigned int)v7 > 1 )
  {
    sub_145E660(a1, v7);
    v7 = *((unsigned int *)a2 + 2);
  }
  v8 = *a2;
  result = v8 + 104 * v7;
  v26 = result;
  if ( v8 != result )
  {
    while ( 1 )
    {
      v27 = *(_QWORD *)v8;
      if ( *(_DWORD *)(v8 + 60) != *(_DWORD *)(v8 + 64) )
        break;
      v10 = *(_QWORD *)(v8 + 8);
      v11 = *(_DWORD *)(a1 + 8);
      v12 = 0;
      if ( v11 >= *(_DWORD *)(a1 + 12) )
        goto LABEL_18;
LABEL_6:
      v13 = (__int64 *)(*(_QWORD *)a1 + 24LL * v11);
      if ( v13 )
      {
        result = v27;
        v13[1] = v10;
        v13[2] = v12;
        *v13 = v27;
        ++*(_DWORD *)(a1 + 8);
LABEL_8:
        v8 += 104;
        if ( v26 == v8 )
          return result;
      }
      else
      {
        result = v11 + 1;
        *(_DWORD *)(a1 + 8) = result;
        if ( !v12 )
          goto LABEL_8;
        *(_QWORD *)v12 = &unk_49EC708;
        v20 = *(unsigned int *)(v12 + 208);
        if ( (_DWORD)v20 )
        {
          v21 = *(_QWORD **)(v12 + 192);
          v22 = &v21[7 * v20];
          do
          {
            if ( *v21 != -16 && *v21 != -8 )
            {
              v23 = v21[1];
              if ( (_QWORD *)v23 != v21 + 3 )
                _libc_free(v23);
            }
            v21 += 7;
          }
          while ( v22 != v21 );
        }
        j___libc_free_0(*(_QWORD *)(v12 + 192));
        v24 = *(_QWORD *)(v12 + 40);
        if ( v24 != v12 + 56 )
          _libc_free(v24);
        v8 += 104;
        result = j_j___libc_free_0(v12, 216);
        if ( v26 == v8 )
          return result;
      }
    }
    v14 = sub_22077B0(216);
    v12 = v14;
    if ( v14 )
      sub_14585E0(v14);
    v15 = *(__int64 **)(v8 + 48);
    if ( v15 == *(__int64 **)(v8 + 40) )
      v16 = *(unsigned int *)(v8 + 60);
    else
      v16 = *(unsigned int *)(v8 + 56);
    v17 = &v15[v16];
    if ( v15 != v17 )
    {
      while ( 1 )
      {
        v18 = *v15;
        v19 = v15;
        if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v17 == ++v15 )
          goto LABEL_17;
      }
      if ( v15 != v17 )
      {
        do
        {
          sub_146E690(v12, v18);
          v25 = v19 + 1;
          if ( v19 + 1 == v17 )
            break;
          v18 = *v25;
          for ( ++v19; (unsigned __int64)*v25 >= 0xFFFFFFFFFFFFFFFELL; v19 = v25 )
          {
            if ( v17 == ++v25 )
              goto LABEL_17;
            v18 = *v25;
          }
        }
        while ( v17 != v19 );
      }
    }
LABEL_17:
    v10 = *(_QWORD *)(v8 + 8);
    v11 = *(_DWORD *)(a1 + 8);
    if ( v11 < *(_DWORD *)(a1 + 12) )
      goto LABEL_6;
LABEL_18:
    sub_145E660(a1, 0);
    v11 = *(_DWORD *)(a1 + 8);
    goto LABEL_6;
  }
  return result;
}
