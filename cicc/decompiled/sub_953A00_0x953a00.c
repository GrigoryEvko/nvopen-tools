// Function: sub_953A00
// Address: 0x953a00
//
__int64 __fastcall sub_953A00(__int64 a1, int *a2)
{
  __int64 v3; // rbx
  int v4; // esi
  int v5; // ecx
  __int64 v6; // rax
  __int64 v8; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 )
  {
    v4 = *a2;
    while ( 1 )
    {
      v5 = *(_DWORD *)(v3 + 32);
      v6 = *(_QWORD *)(v3 + 24);
      if ( v5 > v4 )
        v6 = *(_QWORD *)(v3 + 16);
      if ( !v6 )
        break;
      v3 = v6;
    }
    if ( v5 <= v4 )
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
  v5 = *(_DWORD *)(v8 + 32);
  v3 = v8;
LABEL_8:
  if ( v5 < v4 )
    return 0;
  return v3;
}
