// Function: sub_CB73E0
// Address: 0xcb73e0
//
ssize_t __fastcall sub_CB73E0(__int64 a1, void *a2, size_t a3)
{
  __int64 v4; // rdi
  ssize_t result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  ssize_t v9; // r12
  __int64 v10; // r13
  int v11; // eax

  v4 = *(unsigned int *)(a1 + 48);
  result = read(v4, a2, a3);
  v9 = result;
  if ( result < 0 )
  {
    v10 = sub_2241E50(v4, a2, v6, v7, v8);
    v11 = *__errno_location();
    *(_QWORD *)(a1 + 80) = v10;
    *(_DWORD *)(a1 + 72) = v11;
    return v9;
  }
  else
  {
    *(_QWORD *)(a1 + 88) += result;
  }
  return result;
}
