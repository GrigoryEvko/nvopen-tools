// Function: sub_396ED20
// Address: 0x396ed20
//
__int64 __fastcall sub_396ED20(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 (__fastcall *v5)(__int64, __int64, unsigned __int64); // rbx
  unsigned __int64 v6; // rax

  v2 = *(_QWORD *)(a1 + 256);
  v3 = *(_QWORD *)(a2 + 32);
  v4 = *(_QWORD *)(v3 + 24);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v2 + 240LL);
  v6 = sub_38CB470(*(int *)(v3 + 64), *(_QWORD *)(a1 + 248));
  return v5(v2, v4, v6);
}
