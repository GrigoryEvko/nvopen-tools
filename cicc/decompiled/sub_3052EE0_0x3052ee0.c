// Function: sub_3052EE0
// Address: 0x3052ee0
//
__int64 __fastcall sub_3052EE0(_QWORD *a1, __int64 a2, _QWORD **a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  _QWORD *v9; // rcx
  char v10; // di
  unsigned int v12; // eax

  v5 = sub_22077B0(0x30u);
  *(_QWORD *)(v5 + 32) = **a3;
  *(_WORD *)(v5 + 40) = 0;
  v6 = sub_2FEC750(a1, a2, v5 + 32);
  v8 = v6;
  if ( v7 )
  {
    v9 = a1 + 1;
    v10 = 1;
    if ( !v6 && (_QWORD *)v7 != v9 )
    {
      v12 = *(_DWORD *)(v7 + 32);
      if ( *(_DWORD *)(v5 + 32) >= v12 )
      {
        v10 = 0;
        if ( *(_DWORD *)(v5 + 32) == v12 )
          v10 = *(_WORD *)(v5 + 36) < *(_WORD *)(v7 + 36);
      }
    }
    sub_220F040(v10, v5, (_QWORD *)v7, v9);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v8;
  }
}
