// Function: sub_2F7B080
// Address: 0x2f7b080
//
__int64 __fastcall sub_2F7B080(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int64 v6; // r15
  __int64 v7; // rsi
  _QWORD *v8; // rbx
  _QWORD *v9; // r14
  unsigned __int64 v10; // rdi

  v4 = sub_22077B0(0x28u);
  v5 = v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)v4 = 0;
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)(v4 + 16) = 0;
    *(_DWORD *)(v4 + 24) = 0;
  }
  v6 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v4;
  if ( v6 )
  {
    v7 = *(unsigned int *)(v6 + 24);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD **)(v6 + 8);
      v9 = &v8[4 * v7];
      do
      {
        if ( *v8 != -8192 && *v8 != -4096 )
        {
          v10 = v8[1];
          if ( v10 )
            j_j___libc_free_0(v10);
        }
        v8 += 4;
      }
      while ( v9 != v8 );
      v7 = *(unsigned int *)(v6 + 24);
    }
    sub_C7D6A0(*(_QWORD *)(v6 + 8), 32 * v7, 8);
    j_j___libc_free_0(v6);
    v5 = *(_QWORD *)(a1 + 176);
  }
  return sub_2F7B040(v5, a2);
}
