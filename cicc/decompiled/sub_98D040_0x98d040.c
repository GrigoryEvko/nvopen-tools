// Function: sub_98D040
// Address: 0x98d040
//
__int64 __fastcall sub_98D040(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 v5; // rbx
  __int64 v6; // r12
  char *v8; // rdi

  v2 = *(__int64 **)(a2 + 32);
  v3 = *v2;
  if ( *(_QWORD *)(a1 + 40) != *v2 )
    return 0;
  v5 = *(_QWORD *)(v3 + 56);
  v6 = v3 + 48;
  if ( v5 == v6 )
LABEL_11:
    BUG();
  while ( 1 )
  {
    if ( !v5 )
    {
      v8 = 0;
      goto LABEL_6;
    }
    v8 = (char *)(v5 - 24);
    if ( a1 == v5 - 24 )
      return 1;
LABEL_6:
    if ( !(unsigned __int8)sub_98CD80(v8) )
      return 0;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v6 == v5 )
      goto LABEL_11;
  }
}
