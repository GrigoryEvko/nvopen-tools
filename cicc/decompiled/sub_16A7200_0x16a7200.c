// Function: sub_16A7200
// Address: 0x16a7200
//
__int64 __fastcall sub_16A7200(__int64 a1, __int64 *a2)
{
  __int64 v3; // rcx
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *a2;
  v5 = *(_QWORD *)a1;
  if ( (unsigned int)v3 <= 0x40 )
  {
    *(_QWORD *)a1 = v4 + v5;
    v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
LABEL_3:
    *(_QWORD *)a1 &= v6;
    return a1;
  }
  sub_16A7190(v5, v4, 0, (unsigned __int64)(v3 + 63) >> 6);
  v8 = *(unsigned int *)(a1 + 8);
  v6 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a1 + 8);
  if ( (unsigned int)v8 <= 0x40 )
    goto LABEL_3;
  v9 = (unsigned int)((unsigned __int64)(v8 + 63) >> 6) - 1;
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v9) &= v6;
  return a1;
}
