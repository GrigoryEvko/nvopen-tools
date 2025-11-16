// Function: sub_2A2E770
// Address: 0x2a2e770
//
__int64 __fastcall sub_2A2E770(__int64 a1, unsigned __int16 *a2)
{
  __int64 v3; // rbx
  unsigned __int16 v4; // si
  unsigned __int16 v5; // cx
  __int64 v6; // rax
  __int64 v8; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 )
  {
    v4 = *a2;
    while ( 1 )
    {
      v5 = *(_WORD *)(v3 + 32);
      v6 = *(_QWORD *)(v3 + 24);
      if ( v5 > v4 )
        v6 = *(_QWORD *)(v3 + 16);
      if ( !v6 )
        break;
      v3 = v6;
    }
    if ( v4 >= v5 )
      goto LABEL_8;
  }
  else
  {
    v3 = a1 + 8;
  }
  if ( *(_QWORD *)(a1 + 24) == v3 )
    return 0;
  v8 = sub_220EF80(v3);
  v4 = *a2;
  v5 = *(_WORD *)(v8 + 32);
  v3 = v8;
LABEL_8:
  if ( v4 > v5 )
    return 0;
  return v3;
}
