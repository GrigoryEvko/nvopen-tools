// Function: sub_235A870
// Address: 0x235a870
//
unsigned __int64 __fastcall sub_235A870(unsigned __int64 *a1, unsigned __int64 *a2)
{
  char *v3; // rsi
  unsigned __int64 result; // rax

  v3 = (char *)a1[1];
  if ( v3 == (char *)a1[2] )
    return sub_235A6C0(a1, v3, a2);
  if ( v3 )
  {
    result = *a2;
    *(_QWORD *)v3 = *a2;
    *a2 = 0;
    v3 = (char *)a1[1];
  }
  a1[1] = (unsigned __int64)(v3 + 8);
  return result;
}
