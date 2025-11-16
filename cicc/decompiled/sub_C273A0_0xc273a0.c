// Function: sub_C273A0
// Address: 0xc273a0
//
__int64 __fastcall sub_C273A0(_QWORD *a1, unsigned int *a2)
{
  unsigned int *v2; // r13
  _QWORD *v4; // rcx
  unsigned __int64 v5; // rdi
  _DWORD *v6; // r8
  _DWORD *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  _QWORD *v10; // r14
  unsigned int v11; // edx
  __int64 v12; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  _BOOL8 v19; // rdi
  __int64 v20; // rdi
  unsigned int v21; // eax

  v2 = a2;
  v4 = (_QWORD *)a1[21];
  if ( v4 )
  {
    v5 = v4[1];
    v6 = *(_DWORD **)(*v4 + 8 * (*(_QWORD *)a2 % v5));
    if ( v6 )
    {
      v7 = *(_DWORD **)v6;
      v8 = *(_QWORD *)(*(_QWORD *)v6 + 24LL);
      while ( *(_QWORD *)a2 != v8 || *a2 != v7[2] || a2[1] != v7[3] )
      {
        if ( !*(_QWORD *)v7 )
          goto LABEL_12;
        v8 = *(_QWORD *)(*(_QWORD *)v7 + 24LL);
        v6 = v7;
        if ( *(_QWORD *)a2 % v5 != v8 % v5 )
          goto LABEL_12;
        v7 = *(_DWORD **)v7;
      }
      if ( *(_QWORD *)v6 )
        v2 = (unsigned int *)(*(_QWORD *)v6 + 16LL);
    }
  }
LABEL_12:
  v9 = a1[17];
  v10 = a1 + 16;
  if ( v9 )
  {
    v11 = *v2;
    v12 = (__int64)(a1 + 16);
    while ( 1 )
    {
      while ( *(_DWORD *)(v9 + 32) < v11 )
      {
        v9 = *(_QWORD *)(v9 + 24);
LABEL_18:
        if ( !v9 )
        {
LABEL_19:
          if ( v10 == (_QWORD *)v12
            || *(_DWORD *)(v12 + 32) > v11
            || *(_DWORD *)(v12 + 32) == v11 && v2[1] < *(_DWORD *)(v12 + 36) )
          {
            goto LABEL_26;
          }
          return v12 + 40;
        }
      }
      if ( *(_DWORD *)(v9 + 32) == v11 && *(_DWORD *)(v9 + 36) < v2[1] )
      {
        v9 = *(_QWORD *)(v9 + 24);
        goto LABEL_18;
      }
      v12 = v9;
      v9 = *(_QWORD *)(v9 + 16);
      if ( !v9 )
        goto LABEL_19;
    }
  }
  v12 = (__int64)(a1 + 16);
LABEL_26:
  v14 = v12;
  v12 = sub_22077B0(88);
  v15 = *(_QWORD *)v2;
  *(_DWORD *)(v12 + 48) = 0;
  *(_QWORD *)(v12 + 32) = v15;
  *(_QWORD *)(v12 + 56) = 0;
  *(_QWORD *)(v12 + 64) = v12 + 48;
  *(_QWORD *)(v12 + 72) = v12 + 48;
  *(_QWORD *)(v12 + 80) = 0;
  v16 = sub_C1D380(a1 + 15, v14, (unsigned int *)(v12 + 32));
  v18 = v16;
  if ( v17 )
  {
    v19 = 1;
    if ( !v16 && v10 != (_QWORD *)v17 )
    {
      v21 = *(_DWORD *)(v17 + 32);
      if ( *(_DWORD *)(v12 + 32) >= v21 )
      {
        v19 = 0;
        if ( *(_DWORD *)(v12 + 32) == v21 )
          v19 = *(_DWORD *)(v12 + 36) < *(_DWORD *)(v17 + 36);
      }
    }
    sub_220F040(v19, v12, v17, a1 + 16);
    ++a1[20];
  }
  else
  {
    sub_C1F480(0);
    v20 = v12;
    v12 = v18;
    j_j___libc_free_0(v20, 88);
  }
  return v12 + 40;
}
