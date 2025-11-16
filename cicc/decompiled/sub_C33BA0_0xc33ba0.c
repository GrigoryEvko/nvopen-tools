// Function: sub_C33BA0
// Address: 0xc33ba0
//
bool __fastcall sub_C33BA0(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // eax
  unsigned int v3; // ebx
  _QWORD *v4; // rax
  bool result; // al
  char v6; // cl

  v1 = sub_C33930(a1);
  v2 = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) + 63) >> 6;
  if ( !v2 )
    v2 = 1;
  v3 = v2 - 1;
  if ( v2 == 1 )
  {
LABEL_9:
    v6 = sub_C337A0(a1);
    result = 1;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) > 1u )
      return (*(_QWORD *)(v1 + 8LL * v3) & (0xFFFFFFFFFFFFFFFFLL >> v6)) == 0;
  }
  else
  {
    v4 = (_QWORD *)v1;
    while ( !*v4 )
    {
      if ( ++v4 == (_QWORD *)(v1 + 8LL * v3) )
        goto LABEL_9;
    }
    return 0;
  }
  return result;
}
