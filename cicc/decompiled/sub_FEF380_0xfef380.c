// Function: sub_FEF380
// Address: 0xfef380
//
bool __fastcall sub_FEF380(__int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdi
  _QWORD *v4; // rdx
  _QWORD *v5; // rax
  bool result; // al
  int v7; // edx

  v2 = a2[1];
  v3 = *a2;
  v4 = *(_QWORD **)(v2 + 8);
  if ( v4 && (v5 = *(_QWORD **)(v3 + 8), v4 != v5) )
  {
    while ( v5 )
    {
      v5 = (_QWORD *)*v5;
      if ( v4 == v5 )
        goto LABEL_8;
    }
    return 1;
  }
  else
  {
LABEL_8:
    v7 = *(_DWORD *)(v2 + 16);
    result = 0;
    if ( v7 != -1 )
      return *(_DWORD *)(v3 + 16) != v7;
  }
  return result;
}
