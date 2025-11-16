// Function: sub_39416E0
// Address: 0x39416e0
//
__int64 __fastcall sub_39416E0(_QWORD *a1, __int64 a2, __int64 **a3)
{
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // rcx
  char v12; // di
  unsigned int v14; // eax

  v5 = sub_22077B0(0x50u);
  v6 = *a3;
  *(_QWORD *)(v5 + 40) = 0;
  v7 = *v6;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 32) = v7;
  *(_QWORD *)(v5 + 64) = 0x1000000000LL;
  v8 = sub_3941580(a1, a2, (unsigned int *)(v5 + 32));
  v10 = v8;
  if ( v9 )
  {
    v11 = a1 + 1;
    v12 = 1;
    if ( !v8 && (_QWORD *)v9 != v11 )
    {
      v14 = *(_DWORD *)(v9 + 32);
      if ( *(_DWORD *)(v5 + 32) >= v14 )
      {
        v12 = 0;
        if ( *(_DWORD *)(v5 + 32) == v14 )
          v12 = *(_DWORD *)(v5 + 36) < *(_DWORD *)(v9 + 36);
      }
    }
    sub_220F040(v12, v5, (_QWORD *)v9, v11);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v10;
  }
}
