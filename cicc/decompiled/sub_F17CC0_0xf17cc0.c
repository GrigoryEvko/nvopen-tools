// Function: sub_F17CC0
// Address: 0xf17cc0
//
unsigned __int8 *__fastcall sub_F17CC0(const __m128i *a1, __int64 a2, __int64 a3)
{
  _QWORD *i; // rbx
  _BYTE *v6; // rdi
  unsigned int v7; // r15d
  _QWORD *v9; // rbx
  _BYTE *v10; // rdi
  unsigned int v11; // r15d
  unsigned __int8 *result; // rax

  for ( i = (_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))); (_QWORD *)a2 != i; i += 4 )
  {
    v6 = (_BYTE *)*i;
    if ( *(_BYTE *)*i != 17 )
      goto LABEL_15;
    v7 = *((_DWORD *)v6 + 8);
    if ( !(v7 <= 0x40 ? *((_QWORD *)v6 + 3) == 0 : v7 == (unsigned int)sub_C444A0((__int64)(v6 + 24))) )
      goto LABEL_15;
  }
  v9 = (_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  if ( (_QWORD *)a3 == v9 )
  {
LABEL_15:
    result = (unsigned __int8 *)sub_F0B4D0(a2, a3, (__int64)a1);
    if ( !result )
      return sub_F16D60(a1, a2, a3);
    return result;
  }
  while ( 1 )
  {
    v10 = (_BYTE *)*v9;
    if ( *(_BYTE *)*v9 != 17 )
      break;
    v11 = *((_DWORD *)v10 + 8);
    if ( !(v11 <= 0x40 ? *((_QWORD *)v10 + 3) == 0 : v11 == (unsigned int)sub_C444A0((__int64)(v10 + 24))) )
      break;
    v9 += 4;
    if ( (_QWORD *)a3 == v9 )
      goto LABEL_15;
  }
  result = *(unsigned __int8 **)(a3 + 16);
  if ( result )
  {
    if ( *((_QWORD *)result + 1) )
      return 0;
    goto LABEL_15;
  }
  return result;
}
