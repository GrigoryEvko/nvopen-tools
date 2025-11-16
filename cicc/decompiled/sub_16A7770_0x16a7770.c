// Function: sub_16A7770
// Address: 0x16a7770
//
__int64 __fastcall sub_16A7770(__int64 a1)
{
  unsigned int v2; // ecx
  unsigned __int64 *v3; // rdi
  unsigned __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(unsigned __int64 **)a1;
  if ( v2 <= 0x40 )
  {
    *(_QWORD *)a1 = (char *)v3 - 1;
    v4 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
LABEL_3:
    *(_QWORD *)a1 &= v4;
    return a1;
  }
  sub_16A7730(v3, 1u, ((unsigned __int64)v2 + 63) >> 6);
  v6 = *(unsigned int *)(a1 + 8);
  v4 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a1 + 8);
  if ( (unsigned int)v6 <= 0x40 )
    goto LABEL_3;
  v7 = (unsigned int)((unsigned __int64)(v6 + 63) >> 6) - 1;
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v7) &= v4;
  return a1;
}
