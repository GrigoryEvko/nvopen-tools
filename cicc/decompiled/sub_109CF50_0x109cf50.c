// Function: sub_109CF50
// Address: 0x109cf50
//
bool __fastcall sub_109CF50(_QWORD *a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // rax
  __int64 v5; // rax
  _QWORD *v6; // rcx
  _BYTE *v7; // rax
  __int64 v8; // rdx

  v3 = *(_QWORD *)(a2 + 16);
  if ( !v3 || *(_QWORD *)(v3 + 8) || *(_BYTE *)a2 != 59 )
    return 0;
  v5 = *(_QWORD *)(a2 - 64);
  v6 = (_QWORD *)a1[1];
  if ( v5 )
  {
    *v6 = v5;
    v7 = *(_BYTE **)(a2 - 32);
    if ( a3 == v7 )
      goto LABEL_9;
  }
  else
  {
    v7 = *(_BYTE **)(a2 - 32);
  }
  if ( !v7 )
    return 0;
  *v6 = v7;
  if ( a3 != *(_BYTE **)(a2 - 64) )
    return 0;
LABEL_9:
  if ( *a3 != 69 )
    return 0;
  v8 = *((_QWORD *)a3 - 4);
  if ( !v8 )
    return 0;
  *(_QWORD *)*a1 = v8;
  return (unsigned int)sub_BCB060(*(_QWORD *)(*(_QWORD *)*a1 + 8LL)) == 1;
}
