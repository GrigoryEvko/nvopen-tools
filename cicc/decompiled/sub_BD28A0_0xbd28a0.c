// Function: sub_BD28A0
// Address: 0xbd28a0
//
__int64 __fastcall sub_BD28A0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax

  result = *a1;
  if ( *a1 != *a2 )
  {
    *a1 = *a2;
    v3 = a2[1];
    *a2 = result;
    v4 = a1[1];
    a1[1] = v3;
    v5 = a2[2];
    a2[1] = v4;
    v6 = a1[2];
    a1[2] = v5;
    a2[2] = v6;
    *(_QWORD *)a1[2] = a1;
    v7 = a1[1];
    if ( v7 )
      *(_QWORD *)(v7 + 16) = a1 + 1;
    *(_QWORD *)a2[2] = a2;
    result = a2[1];
    if ( result )
      *(_QWORD *)(result + 16) = a2 + 1;
  }
  return result;
}
