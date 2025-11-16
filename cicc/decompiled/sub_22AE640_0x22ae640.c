// Function: sub_22AE640
// Address: 0x22ae640
//
unsigned __int64 __fastcall sub_22AE640(unsigned __int64 a1)
{
  unsigned __int64 v1; // rax

  v1 = 0x9DDFEA08EB382D69LL
     * (((0x9DDFEA08EB382D69LL * (HIDWORD(a1) ^ (((8 * a1) & 0x7FFFFFFF8LL) + 12995744))) >> 47)
      ^ HIDWORD(a1)
      ^ (0x9DDFEA08EB382D69LL * (HIDWORD(a1) ^ (((8 * a1) & 0x7FFFFFFF8LL) + 12995744))));
  return 0x9DDFEA08EB382D69LL * ((v1 >> 47) ^ v1);
}
