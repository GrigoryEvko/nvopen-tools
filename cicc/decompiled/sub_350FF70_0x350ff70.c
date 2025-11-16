// Function: sub_350FF70
// Address: 0x350ff70
//
_QWORD *__fastcall sub_350FF70(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  int v4; // esi
  int v5; // ecx
  _QWORD *v6; // rax
  __int64 v8; // rdx

  v3 = *(_QWORD **)(a1 + 16);
  if ( v3 )
  {
    v4 = *(_DWORD *)(*(_QWORD *)a2 + 24LL);
    while ( 1 )
    {
      v5 = *(_DWORD *)(v3[4] + 24LL);
      v6 = (_QWORD *)v3[3];
      if ( v5 > v4 )
        v6 = (_QWORD *)v3[2];
      if ( !v6 )
        break;
      v3 = v6;
    }
    if ( v5 <= v4 )
      goto LABEL_8;
  }
  else
  {
    v3 = (_QWORD *)(a1 + 8);
  }
  if ( *(_QWORD **)(a1 + 24) == v3 )
    return 0;
  v8 = sub_220EF80((__int64)v3);
  v4 = *(_DWORD *)(*(_QWORD *)a2 + 24LL);
  v5 = *(_DWORD *)(*(_QWORD *)(v8 + 32) + 24LL);
  v3 = (_QWORD *)v8;
LABEL_8:
  if ( v4 > v5 )
    return 0;
  return v3;
}
