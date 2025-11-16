// Function: sub_E8F2A0
// Address: 0xe8f2a0
//
unsigned __int64 __fastcall sub_E8F2A0(unsigned int *a1)
{
  unsigned int v1; // ecx
  unsigned __int64 v2; // rdx
  __int64 v3; // rsi
  unsigned __int64 result; // rax
  int v5; // eax

  v1 = *a1;
  v2 = *((_QWORD *)a1 + 1);
  v3 = *((_QWORD *)a1 + 2);
  while ( 1 )
  {
    result = *((_QWORD *)a1 - 2);
    if ( v2 >= result && (v2 != result || v1 >= *(a1 - 6)) )
      break;
    *((_QWORD *)a1 + 1) = result;
    v5 = *(a1 - 6);
    a1 -= 6;
    a1[6] = v5;
    *((_QWORD *)a1 + 5) = *((_QWORD *)a1 + 2);
  }
  *((_QWORD *)a1 + 1) = v2;
  *a1 = v1;
  *((_QWORD *)a1 + 2) = v3;
  return result;
}
