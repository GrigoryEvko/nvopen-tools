// Function: sub_C43FB0
// Address: 0xc43fb0
//
unsigned __int64 __fastcall sub_C43FB0(unsigned __int64 *a1, __int64 a2, unsigned int a3, int a4)
{
  unsigned __int64 v7; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v9; // r9
  __int64 v10; // rdx
  char v11; // r11
  __int64 v12; // rdi
  __int64 v13; // rdx

  if ( a4 )
  {
    v7 = *a1;
    result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a4);
    v9 = result & a2;
    if ( *((_DWORD *)a1 + 2) > 0x40u )
      goto LABEL_3;
LABEL_7:
    result <<= a3;
    *a1 = (v9 << a3) | v7 & ~result;
    return result;
  }
  v9 = 0;
  result = 0;
  v7 = *a1;
  if ( *((_DWORD *)a1 + 2) <= 0x40u )
    goto LABEL_7;
LABEL_3:
  v10 = (a3 + a4 - 1) >> 6;
  v11 = a3 & 0x3F;
  v12 = 8LL * (a3 >> 6);
  *(_QWORD *)(v7 + v12) &= ~(result << a3);
  *(_QWORD *)(*a1 + v12) |= v9 << a3;
  if ( (_DWORD)v10 != a3 >> 6 )
  {
    v13 = 8 * v10;
    result = ~(result >> (64 - v11));
    *(_QWORD *)(v13 + *a1) &= result;
    *(_QWORD *)(*a1 + v13) |= v9 >> (64 - v11);
  }
  return result;
}
