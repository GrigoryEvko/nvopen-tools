// Function: sub_AF52E0
// Address: 0xaf52e0
//
bool __fastcall sub_AF52E0(int *a1, __int64 a2)
{
  int v2; // r13d
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r14
  int v8; // r14d
  __int64 v9; // r14
  __int64 v10; // r14
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // r14
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r14
  __int64 v18; // r14

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
  if ( a1[17] != *(_DWORD *)(a2 + 20) )
    return 0;
  v9 = *((_QWORD *)a1 + 9);
  if ( v9 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4) )
    return 0;
  if ( a1[20] != *(_DWORD *)(a2 + 44) )
    return 0;
  v10 = *((_QWORD *)a1 + 11);
  if ( v10 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 5) )
    return 0;
  v11 = *((_QWORD *)a1 + 12);
  if ( v11 != *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6) )
    return 0;
  if ( *((_QWORD *)a1 + 13) == sub_AF5140(a2, 7u)
    && (v12 = *((_QWORD *)a1 + 14), v12 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 8))
    && (v13 = *((_QWORD *)a1 + 15), v13 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 9))
    && (v14 = *((_QWORD *)a1 + 16), v14 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 10))
    && (v15 = *((_QWORD *)a1 + 17), v15 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 11))
    && (v16 = *((_QWORD *)a1 + 18), v16 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 12))
    && (v17 = *((_QWORD *)a1 + 19), v17 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 13))
    && (v18 = *((_QWORD *)a1 + 20), v18 == *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 14)) )
  {
    return a1[42] == *(_DWORD *)(a2 + 40);
  }
  else
  {
    return 0;
  }
}
