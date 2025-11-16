// Function: sub_104B4A0
// Address: 0x104b4a0
//
unsigned __int8 *__fastcall sub_104B4A0(unsigned __int8 **a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned __int8 *result; // rax

  if ( !a4
    || (!a3 ? (v8 = 0, v9 = 0) : (v8 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1), v9 = v8),
        v9 >= *(_DWORD *)(a4 + 32) || !*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v8)) )
  {
LABEL_10:
    *a1 = 0;
    return 0;
  }
  result = sub_104A960((__int64)a1, *a1, a2, a3, a4);
  *a1 = result;
  if ( a5 && result && *result > 0x1Cu )
  {
    if ( (unsigned __int8)sub_B19720(a4, *((_QWORD *)result + 5), a3) )
      return *a1;
    goto LABEL_10;
  }
  return result;
}
