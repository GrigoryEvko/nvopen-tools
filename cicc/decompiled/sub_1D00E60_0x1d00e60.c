// Function: sub_1D00E60
// Address: 0x1d00e60
//
void *__fastcall sub_1D00E60(_QWORD *a1)
{
  void *result; // rax
  _BYTE *v2; // rdx
  _BYTE *v3; // rdi

  a1[6] = 0;
  result = (void *)a1[12];
  if ( result != (void *)a1[13] )
    a1[13] = result;
  v2 = (_BYTE *)a1[16];
  v3 = (_BYTE *)a1[15];
  if ( v3 != v2 )
    return memset(v3, 0, v2 - v3);
  return result;
}
