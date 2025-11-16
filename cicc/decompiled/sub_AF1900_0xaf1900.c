// Function: sub_AF1900
// Address: 0xaf1900
//
char __fastcall sub_AF1900(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // r13
  unsigned __int8 v3; // al
  __int64 v4; // rcx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r14
  int v9; // r14d
  __int64 v10; // rbx
  _BYTE *v11; // rax
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  _BYTE *v14; // rax

  v2 = (_BYTE *)(a2 - 16);
  v3 = *(_BYTE *)(a2 - 16);
  v4 = *a1;
  if ( (v3 & 2) != 0 )
  {
    if ( v4 != *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL) )
      return 0;
  }
  else if ( v4 != *(_QWORD *)(a2 - 8LL * ((v3 >> 2) & 0xF)) )
  {
    return 0;
  }
  v6 = a1[1];
  v7 = a2;
  if ( *(_BYTE *)a2 != 16 )
    v7 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  if ( v6 != v7 )
    return 0;
  if ( *((_DWORD *)a1 + 4) != *(_DWORD *)(a2 + 16) )
    return 0;
  v8 = a1[3];
  if ( v8 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1) )
    return 0;
  if ( a1[4] != *(_QWORD *)(a2 + 24) )
    return 0;
  v9 = *((_DWORD *)a1 + 10);
  if ( v9 != (unsigned int)sub_AF18D0(a2) )
    return 0;
  if ( *((_DWORD *)a1 + 11) != *(_DWORD *)(a2 + 20) )
    return 0;
  v10 = a1[6];
  if ( v10 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 3) )
    return 0;
  v11 = sub_A17150((_BYTE *)(a2 - 16));
  if ( !sub_AF13E0(a1[7], *((_QWORD *)v11 + 4)) )
    return 0;
  v12 = sub_A17150(v2);
  if ( !sub_AF13E0(a1[8], *((_QWORD *)v12 + 5)) )
    return 0;
  v13 = sub_A17150(v2);
  if ( !sub_AF13E0(a1[9], *((_QWORD *)v13 + 6)) )
    return 0;
  v14 = sub_A17150(v2);
  return sub_AF13E0(a1[10], *((_QWORD *)v14 + 7));
}
