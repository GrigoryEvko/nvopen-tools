// Function: ctor_035
// Address: 0x48ca80
//
int ctor_035()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F836E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8375C = 1;
  qword_4F83730 = 0x100000000LL;
  dword_4F836EC &= 0x8000u;
  qword_4F836F8 = 0;
  qword_4F83700 = 0;
  qword_4F83708 = 0;
  dword_4F836E8 = v0;
  word_4F836F0 = 0;
  qword_4F83710 = 0;
  qword_4F83718 = 0;
  qword_4F83720 = 0;
  qword_4F83728 = (__int64)&unk_4F83738;
  qword_4F83740 = 0;
  qword_4F83748 = (__int64)&unk_4F83760;
  qword_4F83750 = 1;
  dword_4F83758 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F83730;
  v3 = (unsigned int)qword_4F83730 + 1LL;
  if ( v3 > HIDWORD(qword_4F83730) )
  {
    sub_C8D5F0((char *)&unk_4F83738 - 16, &unk_4F83738, v3, 8);
    v2 = (unsigned int)qword_4F83730;
  }
  *(_QWORD *)(qword_4F83728 + 8 * v2) = v1;
  LODWORD(qword_4F83730) = qword_4F83730 + 1;
  qword_4F83768 = 0;
  qword_4F83770 = (__int64)&unk_49D9748;
  qword_4F83778 = 0;
  qword_4F836E0 = (__int64)&unk_49DC090;
  qword_4F83780 = (__int64)&unk_49DC1D0;
  qword_4F837A0 = (__int64)nullsub_23;
  qword_4F83798 = (__int64)sub_984030;
  sub_C53080(&qword_4F836E0, "verify-noalias-scope-decl-dom", 29);
  LOBYTE(qword_4F83768) = 0;
  qword_4F83710 = 88;
  LOBYTE(dword_4F836EC) = dword_4F836EC & 0x9F | 0x20;
  LOWORD(qword_4F83778) = 256;
  qword_4F83708 = (__int64)"Ensure that llvm.experimental.noalias.scope.decl for identical scopes are not dominating";
  sub_C53130(&qword_4F836E0);
  return __cxa_atexit(sub_984900, &qword_4F836E0, &qword_4A427C0);
}
