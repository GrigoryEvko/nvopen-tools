// Function: sub_CF97C0
// Address: 0xcf97c0
//
unsigned __int64 __fastcall sub_CF97C0(unsigned int *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax

  v1 = *a1;
  v2 = v1 ^ (unsigned __int64)sub_C64CA0;
  v3 = 0x9DDFEA08EB382D69LL * (v1 ^ (unsigned __int64)sub_C64CA0 ^ (8 * v1 + 4));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v3 ^ v2 ^ (v3 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v3 ^ v2 ^ (v3 >> 47))));
}
