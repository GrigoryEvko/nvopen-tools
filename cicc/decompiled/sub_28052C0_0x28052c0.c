// Function: sub_28052C0
// Address: 0x28052c0
//
unsigned __int64 __fastcall sub_28052C0(unsigned int *a1, unsigned int *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  v2 = *a2 ^ 0xC64CA0LL;
  v3 = 0x9DDFEA08EB382D69LL * (v2 ^ (8LL * *a1 + 8));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v3 >> 47) ^ v3 ^ v2)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v3 >> 47) ^ v3 ^ v2)));
}
