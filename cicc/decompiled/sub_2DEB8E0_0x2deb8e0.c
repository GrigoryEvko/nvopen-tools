// Function: sub_2DEB8E0
// Address: 0x2deb8e0
//
void __fastcall sub_2DEB8E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  unsigned int v4; // r14d
  int v6; // edx
  bool v7; // cf
  int v8; // eax
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // eax
  __int64 v12; // rdx
  int *v13; // r12
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rdx
  int v18; // eax
  bool v19; // cc
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // r13
  unsigned __int64 v24; // rdi
  __int64 v25; // rdi
  char *v26; // r12
  int v27; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v28; // [rsp+8h] [rbp-38h] BYREF
  unsigned int v29; // [rsp+10h] [rbp-30h]

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 != *(_DWORD *)(a1 + 136) )
  {
    *(_DWORD *)a1 = -1;
    return;
  }
  if ( v3 <= 0x40 )
  {
    if ( *(_QWORD *)a2 == 1 )
      return;
    _RAX = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 )
      goto LABEL_24;
    goto LABEL_28;
  }
  if ( (unsigned int)sub_C444A0(a2) == v3 - 1 )
    return;
  if ( v3 == (unsigned int)sub_C444A0(a2) )
  {
LABEL_28:
    v21 = *(unsigned int *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    v23 = v22 + 24 * v21;
    if ( v22 == v23 )
    {
      v4 = 0;
    }
    else
    {
      do
      {
        v23 -= 24;
        if ( *(_DWORD *)(v23 + 16) > 0x40u )
        {
          v24 = *(_QWORD *)(v23 + 8);
          if ( v24 )
            j_j___libc_free_0_0(v24);
        }
      }
      while ( v22 != v23 );
      v4 = *(_DWORD *)a1;
    }
    *(_DWORD *)(a1 + 24) = 0;
    v3 = *(_DWORD *)(a2 + 8);
    if ( v3 > 0x40 )
      goto LABEL_8;
    _RAX = *(_QWORD *)a2;
    if ( !*(_QWORD *)a2 )
    {
      LODWORD(_RAX) = 64;
LABEL_25:
      v4 = *(_DWORD *)a1;
      if ( (unsigned int)_RAX > v3 )
        LODWORD(_RAX) = v3;
      goto LABEL_9;
    }
LABEL_24:
    __asm { tzcnt   rax, rax }
    goto LABEL_25;
  }
  v4 = *(_DWORD *)a1;
LABEL_8:
  LODWORD(_RAX) = sub_C44590(a2);
LABEL_9:
  if ( v4 != -1 )
  {
    v6 = v4 - _RAX;
    v7 = (unsigned int)_RAX < v4;
    v8 = 0;
    if ( v7 )
      v8 = v6;
    *(_DWORD *)a1 = v8;
  }
  sub_C47360(a1 + 128, (__int64 *)a2);
  if ( *(_QWORD *)(a1 + 8) )
  {
    v11 = *(_DWORD *)(a2 + 8);
    v27 = 1;
    v29 = v11;
    if ( v11 > 0x40 )
      sub_C43780((__int64)&v28, (const void **)a2);
    else
      v28 = *(_QWORD *)a2;
    v12 = *(unsigned int *)(a1 + 24);
    v13 = &v27;
    v14 = *(_QWORD *)(a1 + 16);
    v15 = v12 + 1;
    v16 = *(_DWORD *)(a1 + 24);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
    {
      v25 = a1 + 16;
      if ( v14 > (unsigned __int64)&v27 || (unsigned __int64)&v27 >= v14 + 24 * v12 )
      {
        sub_2DEABD0(v25, v15, v12, v14, v9, v10);
        v12 = *(unsigned int *)(a1 + 24);
        v14 = *(_QWORD *)(a1 + 16);
        v16 = *(_DWORD *)(a1 + 24);
      }
      else
      {
        v26 = (char *)&v27 - v14;
        sub_2DEABD0(v25, v15, v12, v14, v9, v10);
        v14 = *(_QWORD *)(a1 + 16);
        v12 = *(unsigned int *)(a1 + 24);
        v13 = (int *)&v26[v14];
        v16 = *(_DWORD *)(a1 + 24);
      }
    }
    v17 = v14 + 24 * v12;
    if ( v17 )
    {
      *(_DWORD *)v17 = *v13;
      v18 = v13[4];
      v13[4] = 0;
      *(_DWORD *)(v17 + 16) = v18;
      *(_QWORD *)(v17 + 8) = *((_QWORD *)v13 + 1);
      v16 = *(_DWORD *)(a1 + 24);
    }
    v19 = v29 <= 0x40;
    *(_DWORD *)(a1 + 24) = v16 + 1;
    if ( !v19 )
    {
      if ( v28 )
        j_j___libc_free_0_0(v28);
    }
  }
}
