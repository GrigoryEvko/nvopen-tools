// Function: sub_2ED7930
// Address: 0x2ed7930
//
__int64 __fastcall sub_2ED7930(__int64 a1, int a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int *v7; // r14
  __int64 v8; // rdx
  __int64 *v9; // r12
  const void *v10; // rax
  const void *v11; // rsi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  void *v14; // r13
  unsigned int v15; // esi
  __int64 v16; // rcx
  unsigned int v17; // r12d
  unsigned int v18; // eax
  unsigned int v19; // r12d
  int v21; // [rsp+1Ch] [rbp-34h]

  v21 = *(_DWORD *)(*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 376LL))(
                     *(_QWORD *)(a1 + 16),
                     a3)
      * a2;
  v7 = (unsigned int *)(*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 416LL))(
                         *(_QWORD *)(a1 + 16),
                         a3);
  v9 = sub_2ED6C00(a1, a4, 1);
  v10 = (const void *)v9[1];
  v11 = (const void *)*v9;
  v12 = (unsigned __int64)v10 - *v9;
  if ( v10 == (const void *)*v9 )
  {
    v14 = 0;
  }
  else
  {
    if ( v12 > 0x7FFFFFFFFFFFFFFCLL )
      sub_4261EA(v12, v11, v8);
    v13 = sub_22077B0(v12);
    v11 = (const void *)*v9;
    v14 = (void *)v13;
    v10 = (const void *)v9[1];
    v12 = (unsigned __int64)v10 - *v9;
  }
  if ( v10 != v11 )
  {
    memmove(v14, v11, v12);
    v15 = *v7;
    if ( *v7 != -1 )
    {
LABEL_6:
      v16 = *(_QWORD *)(a1 + 384);
      while ( 1 )
      {
        v17 = *((_DWORD *)v14 + (int)v15) + v21;
        v18 = *(_DWORD *)(v16 + 4LL * v15);
        if ( v18 )
        {
          if ( v17 >= v18 )
            goto LABEL_11;
        }
        else
        {
          *(_DWORD *)(*(_QWORD *)(a1 + 384) + 4LL * v15) = sub_2F60A40(a1 + 88);
          v16 = *(_QWORD *)(a1 + 384);
          if ( v17 >= *(_DWORD *)(v16 + 4LL * v15) )
          {
LABEL_11:
            v19 = 1;
            goto LABEL_12;
          }
        }
        v15 = v7[1];
        ++v7;
        if ( v15 == -1 )
          goto LABEL_16;
      }
    }
    v19 = 0;
    goto LABEL_13;
  }
  v15 = *v7;
  if ( *v7 != -1 )
    goto LABEL_6;
LABEL_16:
  v19 = 0;
LABEL_12:
  if ( v14 )
LABEL_13:
    j_j___libc_free_0((unsigned __int64)v14);
  return v19;
}
