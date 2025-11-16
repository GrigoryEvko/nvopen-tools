// Function: sub_3937F10
// Address: 0x3937f10
//
__int64 __fastcall sub_3937F10(_QWORD *a1)
{
  __int64 v1; // rdx
  unsigned __int16 *v2; // rax
  __int64 v3; // rcx
  __int64 result; // rax

  v1 = a1[3];
  v2 = (unsigned __int16 *)a1[2];
  if ( !v1 )
  {
    v1 = *v2++;
    a1[3] = v1;
  }
  a1[2] = v2 + 4;
  v3 = *((_QWORD *)v2 + 1);
  a1[2] = v2 + 8;
  result = (__int64)v2 + *((_QWORD *)v2 + 2) + v3 + 24;
  --a1[4];
  a1[2] = result;
  a1[3] = v1 - 1;
  return result;
}
