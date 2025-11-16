// Function: sub_32157B0
// Address: 0x32157b0
//
__int64 __fastcall sub_32157B0(__int64 *a1, __int64 a2, unsigned __int16 a3)
{
  __int64 *v3; // rax
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v7; // [rsp+8h] [rbp-8h] BYREF

  if ( a3 == 14 )
  {
    v3 = (__int64 *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    if ( ((*a1 >> 2) & 1) == 0 )
      ++v3;
    if ( *(_BYTE *)(a2 + 4) )
    {
      v7 = *v3;
      return sub_3215500((__int64)&v7, a2, 14);
    }
    else
    {
      v7 = v3[1];
      return sub_3215240(&v7, (unsigned int *)a2, 0xEu);
    }
  }
  if ( a3 <= 0xDu )
    goto LABEL_16;
  if ( a3 <= 0x28u )
  {
    if ( a3 > 0x24u || a3 == 26 )
      goto LABEL_10;
LABEL_16:
    BUG();
  }
  if ( a3 != 7938 )
    goto LABEL_16;
LABEL_10:
  v5 = *a1;
  v6 = (v5 & 0xFFFFFFFFFFFFFFF8LL) + 8;
  if ( (v5 & 4) != 0 )
    v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = *(unsigned int *)(v6 + 16);
  return sub_3215240(&v7, (unsigned int *)a2, a3);
}
