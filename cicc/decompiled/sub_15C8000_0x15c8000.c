// Function: sub_15C8000
// Address: 0x15c8000
//
__int64 __fastcall sub_15C8000(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // r13
  size_t v7; // rdx
  __int64 v8; // r12
  __int64 (__fastcall *v9)(__int64, __int64, size_t); // r14

  v2 = sub_15E0530(*(_QWORD *)(a1 + 16));
  v5 = sub_16033E0(v2, a2, v3, v4);
  v6 = *(_QWORD *)(a1 + 48);
  v7 = 0;
  v8 = v5;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, size_t))(*(_QWORD *)v5 + 40LL);
  if ( v6 )
    v7 = strlen(*(const char **)(a1 + 48));
  return v9(v8, v6, v7);
}
