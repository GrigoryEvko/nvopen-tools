// Function: sub_888210
// Address: 0x888210
//
__int64 __fastcall sub_888210(int *a1, __int64 *a2)
{
  int v2; // ecx
  int v3; // edx
  __int64 result; // rax
  __int64 v5; // r8

  v2 = *a1;
  v3 = a1[1];
  if ( *(_QWORD *)a1 )
  {
    result = *a2;
    v5 = *((_QWORD *)a1 + 1);
    *(_QWORD *)a1 = *a2;
    if ( result )
    {
      result = a2[1];
      *((_QWORD *)a1 + 1) = result;
    }
    *(_DWORD *)a2 = v2;
    *((_DWORD *)a2 + 1) = v3;
    a2[1] = v5;
  }
  else
  {
    result = *a2;
    *(_QWORD *)a1 = *a2;
    if ( result )
    {
      result = a2[1];
      *((_QWORD *)a1 + 1) = result;
    }
    *a2 = 0;
  }
  return result;
}
