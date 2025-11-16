// Function: sub_CB70C0
// Address: 0xcb70c0
//
__off_t __fastcall sub_CB70C0(__int64 a1, __off_t a2)
{
  __int64 v3; // rdi
  __off_t v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __off_t v8; // r12
  __int64 v9; // r13
  int v10; // eax

  if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a1 + 16) )
    sub_CB5AE0((__int64 *)a1);
  v3 = *(unsigned int *)(a1 + 48);
  v4 = lseek(v3, a2, 0);
  *(_QWORD *)(a1 + 88) = v4;
  v8 = v4;
  if ( v4 == -1 )
  {
    v9 = sub_2241E50(v3, a2, v5, v6, v7);
    v10 = *__errno_location();
    *(_QWORD *)(a1 + 80) = v9;
    *(_DWORD *)(a1 + 72) = v10;
  }
  return v8;
}
