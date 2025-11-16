// Function: sub_1CEC070
// Address: 0x1cec070
//
__int64 __fastcall sub_1CEC070(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  const char **v6; // r13
  const char *v7; // r15
  size_t v8; // rcx

  if ( !sub_1456C80(*a1, *a2) )
    return 0;
  v2 = *a1;
  v3 = sub_146F1B0(*a1, (__int64)a2);
  v4 = sub_1456F20(v2, v3);
  if ( *(_WORD *)(v4 + 24) != 10 )
    return 0;
  v5 = *(_QWORD *)(v4 - 8);
  v6 = (const char **)&unk_4FC0620;
  if ( *(_BYTE *)(v5 + 16) != 78 )
    v5 = 0;
  while ( 1 )
  {
    if ( v5 )
    {
      v7 = *v6;
      v8 = 0;
      if ( *v6 )
        v8 = strlen(*v6);
      if ( sub_1CEBF90(a1, v5, v7, v8) )
        break;
    }
    if ( ++v6 == (const char **)&dword_4FC0638 )
      return 0;
  }
  return v5;
}
