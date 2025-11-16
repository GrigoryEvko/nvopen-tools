// Function: sub_13B83E0
// Address: 0x13b83e0
//
unsigned __int64 __fastcall sub_13B83E0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 *v3; // rdi
  unsigned __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  char v6; // [rsp+10h] [rbp-10h]

  v2 = a1 + 40;
  v3 = (unsigned __int64 *)(a1 + 104);
  *(v3 - 12) = v2;
  *(v3 - 11) = v2;
  *(v3 - 8) = a2;
  *v3 = 0;
  v3[1] = 0;
  v3[2] = 0;
  *(v3 - 10) = 0x100000008LL;
  *((_DWORD *)v3 - 18) = 0;
  *(v3 - 13) = 1;
  v5 = a2;
  v6 = 0;
  return sub_13B8390(v3, (__int64)&v5);
}
