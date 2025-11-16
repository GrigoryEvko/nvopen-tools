// Function: sub_13A4D00
// Address: 0x13a4d00
//
void __fastcall sub_13A4D00(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  int v7; // eax
  __int64 v8; // r14
  __int64 v9; // rax
  size_t v10; // rdx
  void *v11; // r15
  const void *v12; // rsi
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r12
  unsigned __int64 v17; // r15
  void *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  size_t n; // [rsp+8h] [rbp-38h]

  v2 = *a2;
  v3 = *a1;
  v4 = *a2 & 1;
  if ( (*a1 & 1) != 0 )
  {
    if ( (_BYTE)v4 )
    {
LABEL_3:
      *a1 = v2;
      return;
    }
    v5 = (_QWORD *)sub_22077B0(24);
    v6 = v5;
    if ( v5 )
    {
      *v5 = 0;
      v5[1] = 0;
      v7 = *(_DWORD *)(v2 + 16);
      *((_DWORD *)v6 + 4) = v7;
      if ( v7 )
      {
        v8 = (unsigned int)(v7 + 63) >> 6;
        v9 = malloc(8 * v8);
        v10 = 8 * v8;
        v11 = (void *)v9;
        if ( !v9 )
        {
          if ( 8 * v8 || (v19 = malloc(1u), v10 = 0, !v19) )
          {
            n = v10;
            sub_16BD1C0("Allocation failed");
            v10 = n;
          }
          else
          {
            v11 = (void *)v19;
          }
        }
        *v6 = v11;
        v12 = *(const void **)v2;
        v6[1] = v8;
        memcpy(v11, v12, v10);
      }
    }
    *a1 = (__int64)v6;
  }
  else
  {
    if ( (_BYTE)v4 )
    {
      if ( v3 )
      {
        _libc_free(*(_QWORD *)v3);
        j_j___libc_free_0(v3, 24);
        v2 = *a2;
      }
      goto LABEL_3;
    }
    if ( v2 != v3 )
    {
      v13 = *(unsigned int *)(v2 + 16);
      v14 = *(_QWORD *)(v3 + 8) << 6;
      *(_DWORD *)(v3 + 16) = v13;
      v15 = (unsigned int)(v13 + 63) >> 6;
      if ( v13 > v14 )
      {
        v16 = (unsigned int)v15;
        v17 = 8LL * (unsigned int)v15;
        v18 = (void *)malloc(v17);
        if ( !v18 )
        {
          if ( v17 || (v20 = malloc(1u)) == 0 )
            sub_16BD1C0("Allocation failed");
          else
            v18 = (void *)v20;
        }
        memcpy(v18, *(const void **)v2, v17);
        _libc_free(*(_QWORD *)v3);
        *(_QWORD *)v3 = v18;
        *(_QWORD *)(v3 + 8) = v16;
      }
      else
      {
        if ( (_DWORD)v13 )
          memcpy(*(void **)v3, *(const void **)v2, 8 * v15);
        sub_13A4C60(v3, 0);
      }
    }
  }
}
