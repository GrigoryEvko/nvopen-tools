// Function: sub_16A7490
// Address: 0x16a7490
//
__int64 __fastcall sub_16A7490(__int64 a1, __int64 a2)
{
  unsigned int v3; // ecx
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx

  v3 = *(_DWORD *)(a1 + 8);
  v4 = *(_QWORD **)a1;
  if ( v3 <= 0x40 )
  {
    *(_QWORD *)a1 = (char *)v4 + a2;
    v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
LABEL_3:
    *(_QWORD *)a1 &= v5;
    return a1;
  }
  sub_16A73B0(v4, a2, ((unsigned __int64)v3 + 63) >> 6);
  v7 = *(unsigned int *)(a1 + 8);
  v5 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a1 + 8);
  if ( (unsigned int)v7 <= 0x40 )
    goto LABEL_3;
  v8 = (unsigned int)((unsigned __int64)(v7 + 63) >> 6) - 1;
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v8) &= v5;
  return a1;
}
