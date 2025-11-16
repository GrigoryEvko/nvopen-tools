// Function: sub_C339A0
// Address: 0xc339a0
//
__int64 __fastcall sub_C339A0(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // eax
  unsigned int v3; // ebx
  _QWORD *v4; // rax
  __int64 result; // rax
  char v6; // dl

  v1 = sub_C33930(a1);
  v2 = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) + 63) >> 6;
  if ( !v2 )
    v2 = 1;
  v3 = v2 - 1;
  if ( v2 != 1 )
  {
    v4 = (_QWORD *)v1;
    while ( *v4 == -1 )
    {
      if ( ++v4 == (_QWORD *)(v1 + 8LL * v3) )
        goto LABEL_8;
    }
    return 0;
  }
LABEL_8:
  v6 = sub_C337A0(a1);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) <= 1u )
    return 0;
  result = 1;
  if ( (*(_QWORD *)(v1 + 8LL * v3) | (-1LL << (64 - v6))) != 0xFFFFFFFFFFFFFFFFLL )
    return 0;
  return result;
}
