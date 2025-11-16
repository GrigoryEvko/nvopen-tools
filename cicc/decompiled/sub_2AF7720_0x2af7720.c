// Function: sub_2AF7720
// Address: 0x2af7720
//
__int64 __fastcall sub_2AF7720(_QWORD *a1, char **a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // r12
  char *v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // rcx
  char *v7; // rdi
  __int64 result; // rax

  v2 = a1;
  v3 = *a1;
  while ( 1 )
  {
    v4 = *a2;
    v5 = v2;
    v6 = *(v2 - 1);
    v7 = &a2[1][(_QWORD)a2[3]];
    if ( ((unsigned __int8)*a2 & 1) != 0 )
      v4 = *(char **)&v4[*(_QWORD *)v7 - 1];
    --v2;
    result = ((__int64 (__fastcall *)(char *, char *, __int64, __int64))v4)(v7, a2[2], v3, v6);
    if ( !(_BYTE)result )
      break;
    v2[1] = *v2;
  }
  *v5 = v3;
  return result;
}
