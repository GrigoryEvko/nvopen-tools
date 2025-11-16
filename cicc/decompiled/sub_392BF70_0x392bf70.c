// Function: sub_392BF70
// Address: 0x392bf70
//
bool __fastcall sub_392BF70(__int64 a1, const char *a2)
{
  const char *v2; // r13
  size_t v3; // rax

  v2 = *(const char **)(*(_QWORD *)(a1 + 136) + 40LL);
  v3 = strlen(v2);
  return strncmp(a2, v2, v3) == 0;
}
