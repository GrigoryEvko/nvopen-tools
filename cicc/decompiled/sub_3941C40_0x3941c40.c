// Function: sub_3941C40
// Address: 0x3941c40
//
__int64 __fastcall sub_3941C40(_QWORD *a1, __int64 a2, __int64 **a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  char v11; // di
  unsigned int v13; // eax

  v5 = sub_22077B0(0x58u);
  v6 = **a3;
  *(_DWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 64) = v5 + 48;
  *(_QWORD *)(v5 + 72) = v5 + 48;
  *(_QWORD *)(v5 + 80) = 0;
  v7 = sub_3941AE0(a1, a2, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v10 != (_QWORD *)v8 )
    {
      v13 = *(_DWORD *)(v8 + 32);
      if ( *(_DWORD *)(v5 + 32) >= v13 )
      {
        v11 = 0;
        if ( *(_DWORD *)(v5 + 32) == v13 )
          v11 = *(_DWORD *)(v5 + 36) < *(_DWORD *)(v8 + 36);
      }
    }
    sub_220F040(v11, v5, (_QWORD *)v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    sub_393E140(0);
    j_j___libc_free_0(v5);
    return v9;
  }
}
