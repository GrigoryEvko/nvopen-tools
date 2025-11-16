// Function: sub_E3F130
// Address: 0xe3f130
//
__int64 __fastcall sub_E3F130(__int64 a1, char a2)
{
  __int64 v3; // rdi
  char *v4; // rax

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(char **)(v3 + 32);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 24) )
  {
    sub_CB5D20(v3, a2);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v4 + 1;
    *v4 = a2;
  }
  return a1;
}
