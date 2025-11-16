// Function: sub_394A5C0
// Address: 0x394a5c0
//
__int64 __fastcall sub_394A5C0(_QWORD *a1, char *a2, unsigned __int64 a3)
{
  unsigned int v3; // r13d
  size_t v5; // r14
  void *v8; // rax
  char *v9; // rdi
  __int64 v10; // rsi
  unsigned int v11; // ecx
  unsigned __int64 v12; // r10
  _QWORD *v13; // r11
  _QWORD *v14; // rax
  _QWORD *v15; // r9
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 *v18; // r10
  __int64 v19; // r11
  __int64 v20; // r8
  unsigned int *v21; // r9
  unsigned int v22; // eax

  v3 = *(unsigned __int8 *)a1;
  if ( (_BYTE)v3 )
    return 0;
  v5 = a1[2] - a1[1];
  if ( v5 > 0x7FFFFFFFFFFFFFFCLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v5 )
  {
    v8 = (void *)sub_22077B0(a1[2] - a1[1]);
    v9 = (char *)memset(v8, 0, v5);
    if ( !a3 )
    {
      v3 = 1;
LABEL_21:
      j_j___libc_free_0((unsigned __int64)v9);
      return v3;
    }
  }
  else
  {
    v9 = 0;
    if ( !a3 )
      return 1;
  }
  v10 = 0;
  v11 = *a2 & 0xFFFFFF;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
LABEL_8:
        if ( a3 <= ++v10 )
        {
          v3 = 1;
          goto LABEL_20;
        }
        v11 = (a2[v10] + (v11 << 8)) & 0xFFFFFF;
        if ( v10 != 1 )
        {
          v12 = a1[5];
          v13 = *(_QWORD **)(a1[4] + 8 * (v11 % v12));
          if ( v13 )
            break;
        }
      }
      v14 = (_QWORD *)*v13;
      if ( v11 == *(_DWORD *)(*v13 + 8LL) )
        break;
      while ( 1 )
      {
        v15 = (_QWORD *)*v14;
        if ( !*v14 )
          break;
        v13 = v14;
        if ( v11 % v12 != *((unsigned int *)v15 + 2) % v12 )
          break;
        v14 = (_QWORD *)*v14;
        if ( v11 == *((_DWORD *)v15 + 2) )
          goto LABEL_15;
      }
    }
LABEL_15:
    v16 = *v13;
    if ( *v13 )
    {
      v17 = *(__int64 **)(v16 + 16);
      v18 = &v17[*(unsigned int *)(v16 + 24)];
      if ( v17 != v18 )
        break;
    }
  }
  v19 = a1[1];
  while ( 1 )
  {
    v20 = *v17;
    v21 = (unsigned int *)&v9[4 * *v17];
    v22 = *v21 + 1;
    *v21 = v22;
    if ( v22 >= *(_DWORD *)(v19 + 4 * v20) )
      break;
    if ( v18 == ++v17 )
      goto LABEL_8;
  }
LABEL_20:
  if ( v9 )
    goto LABEL_21;
  return v3;
}
