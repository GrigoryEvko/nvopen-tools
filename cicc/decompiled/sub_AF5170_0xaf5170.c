// Function: sub_AF5170
// Address: 0xaf5170
//
bool __fastcall sub_AF5170(int *a1, __int64 a2)
{
  int v2; // r13d
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r14
  int v8; // r14d
  char v9; // al
  char v10; // al
  __int64 v11; // rbx
  __int64 v12; // rbx
  __int64 v13; // [rsp+8h] [rbp-28h]

  v2 = *a1;
  if ( v2 != (unsigned __int16)sub_AF18C0(a2) || *((_QWORD *)a1 + 1) != sub_AF5140(a2, 2u) )
    return 0;
  v4 = *((_QWORD *)a1 + 2);
  v5 = a2;
  if ( *(_BYTE *)a2 != 16 )
    v5 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  if ( v4 != v5 )
    return 0;
  if ( a1[6] != *(_DWORD *)(a2 + 16) )
    return 0;
  v6 = *((_QWORD *)a1 + 4);
  if ( v6 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1) )
    return 0;
  v7 = *((_QWORD *)a1 + 5);
  if ( v7 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 3) )
    return 0;
  if ( *((_QWORD *)a1 + 6) != *(_QWORD *)(a2 + 24) )
    return 0;
  v8 = a1[16];
  if ( v8 != (unsigned int)sub_AF18D0(a2) )
    return 0;
  if ( *((_QWORD *)a1 + 7) != *(_QWORD *)(a2 + 32) )
    return 0;
  v9 = *((_BYTE *)a1 + 72);
  if ( v9 != *(_BYTE *)(a2 + 48) || v9 && a1[17] != *(_DWORD *)(a2 + 44) )
    return 0;
  v13 = sub_AF2E40(a2);
  v10 = *((_BYTE *)a1 + 80);
  if ( v10 != BYTE4(v13) || v10 && a1[19] != (_DWORD)v13 )
    return 0;
  if ( a1[21] != *(_DWORD *)(a2 + 20) )
    return 0;
  v11 = *((_QWORD *)a1 + 11);
  if ( v11 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4) )
    return 0;
  v12 = *((_QWORD *)a1 + 12);
  return *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 5) == v12;
}
