// Function: sub_1290AF0
// Address: 0x1290af0
//
__int64 __fastcall sub_1290AF0(_QWORD *a1, _QWORD *a2, char a3)
{
  unsigned __int64 *v4; // r14
  unsigned __int64 v5; // rcx
  __int64 v6; // rax
  __int64 result; // rax

  sub_12909B0(a1, (__int64)a2);
  if ( a3 && !a2[1] )
  {
    sub_157EF40(a2);
    return j_j___libc_free_0(a2, 64);
  }
  else
  {
    v4 = (unsigned __int64 *)(a1[15] + 72LL);
    sub_15E01D0(v4, a2);
    v5 = *v4;
    v6 = a2[3];
    a2[4] = v4;
    v5 &= 0xFFFFFFFFFFFFFFF8LL;
    a2[3] = v5 | v6 & 7;
    *(_QWORD *)(v5 + 8) = a2 + 3;
    result = *v4 & 7;
    *v4 = result | (unsigned __int64)(a2 + 3);
    a1[7] = a2;
    a1[8] = a2 + 5;
  }
  return result;
}
