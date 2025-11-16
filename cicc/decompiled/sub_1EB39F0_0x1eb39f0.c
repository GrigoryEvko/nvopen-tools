// Function: sub_1EB39F0
// Address: 0x1eb39f0
//
__int64 __fastcall sub_1EB39F0(__int64 *a1, int a2)
{
  __int64 *v2; // r14
  __int64 *v4; // r12
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  _BOOL8 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 *v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-38h]

  v2 = a1 + 10;
  v4 = a1 + 10;
  v6 = a1[11];
  if ( v6 )
  {
    do
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v6 + 16);
        v8 = *(_QWORD *)(v6 + 24);
        if ( *(_DWORD *)(v6 + 32) >= a2 )
          break;
        v6 = *(_QWORD *)(v6 + 24);
        if ( !v8 )
          goto LABEL_6;
      }
      v4 = (__int64 *)v6;
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v7 );
LABEL_6:
    if ( v2 != v4 && *((_DWORD *)v4 + 8) <= a2 )
    {
LABEL_8:
      result = v4[5];
      if ( result )
        return result;
      goto LABEL_15;
    }
  }
  v10 = sub_22077B0(48);
  v11 = (__int64)v4;
  *(_DWORD *)(v10 + 32) = a2;
  v4 = (__int64 *)v10;
  *(_QWORD *)(v10 + 40) = 0;
  v12 = sub_1EB38F0(a1 + 9, v11, (int *)(v10 + 32));
  if ( !v13 )
  {
    v17 = v4;
    v4 = (__int64 *)v12;
    j_j___libc_free_0(v17, 48);
    goto LABEL_8;
  }
  v14 = v2 == (__int64 *)v13 || v12 || a2 < *(_DWORD *)(v13 + 32);
  sub_220F040(v14, v4, v13, v2);
  ++a1[14];
  result = v4[5];
  if ( !result )
  {
LABEL_15:
    v15 = *a1;
    result = sub_22077B0(24);
    if ( result )
    {
      v18 = result;
      sub_1EB3690(result, 4, v15);
      result = v18;
      *(_DWORD *)(v18 + 16) = a2;
      *(_QWORD *)v18 = &unk_49FD7D8;
    }
    v16 = v4[5];
    v4[5] = result;
    if ( v16 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 16LL))(v16);
      return v4[5];
    }
  }
  return result;
}
