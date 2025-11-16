// Function: sub_1E37160
// Address: 0x1e37160
//
__int64 __fastcall sub_1E37160(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r13
  size_t v4; // rdx
  __int64 v5; // r12
  __int64 (__fastcall *v6)(__int64, __int64, size_t); // r14

  v1 = sub_15E0530(*(_QWORD *)(a1 + 16));
  v2 = sub_16033E0(v1);
  v3 = *(_QWORD *)(a1 + 48);
  v4 = 0;
  v5 = v2;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, size_t))(*(_QWORD *)v2 + 40LL);
  if ( v3 )
    v4 = strlen(*(const char **)(a1 + 48));
  return v6(v5, v3, v4);
}
