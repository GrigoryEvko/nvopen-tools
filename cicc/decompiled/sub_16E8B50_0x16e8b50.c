// Function: sub_16E8B50
// Address: 0x16e8b50
//
__off_t __fastcall sub_16E8B50(__int64 a1, __off_t a2)
{
  __int64 v3; // rdi
  __off_t v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __off_t v8; // r12
  __int64 v9; // r13
  int v10; // eax

  if ( *(_QWORD *)(a1 + 24) != *(_QWORD *)(a1 + 8) )
    sub_16E7BA0((__int64 *)a1);
  v3 = *(unsigned int *)(a1 + 36);
  v4 = lseek(v3, a2, 0);
  *(_QWORD *)(a1 + 64) = v4;
  v8 = v4;
  if ( v4 == -1 )
  {
    v9 = sub_2241E50(v3, a2, v5, v6, v7);
    v10 = *__errno_location();
    *(_QWORD *)(a1 + 56) = v9;
    *(_DWORD *)(a1 + 48) = v10;
  }
  return v8;
}
