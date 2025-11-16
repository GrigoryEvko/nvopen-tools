// Function: sub_325FFE0
// Address: 0x325ffe0
//
__int64 __fastcall sub_325FFE0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rdi
  void *v6; // r12
  char v7; // al
  unsigned __int8 v8; // al
  unsigned __int8 v10; // al
  int v11; // eax
  int v12; // edx
  int v13; // r12d
  int v14; // r14d
  int v15; // eax
  int v16; // edx
  __int64 v17; // rax

  if ( !*(_QWORD *)a2 )
    return 0;
  v2 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v3 = *a1;
  v4 = v2 + 24;
  v5 = v2 + 24;
  v6 = sub_C33340();
  if ( *(void **)(v2 + 24) == v6 )
    v7 = sub_C40310(v5);
  else
    v7 = sub_C33940(v5);
  if ( v7 )
    return 0;
  if ( v6 == *(void **)(v2 + 24) )
  {
    v4 = *(_QWORD *)(v2 + 32);
    v10 = *(_BYTE *)(v4 + 20) & 7;
    if ( v10 == 3 || v10 <= 1u )
      return 0;
  }
  else
  {
    v8 = *(_BYTE *)(v2 + 44) & 7;
    if ( v8 <= 1u || v8 == 3 )
      return 0;
  }
  v11 = sub_C3BD20(v4);
  v12 = *(_DWORD *)(v3 + 8);
  v13 = v11;
  v14 = v11;
  v15 = *(_DWORD *)(*(_QWORD *)v3 + 24LL);
  if ( v15 == 98 || (v14 -= v12, v15 != 99) )
    v13 += v12;
  if ( v14 <= (int)sub_C336C0(*(_QWORD *)(v2 + 24)) || (int)sub_C336B0(*(unsigned int **)(v2 + 24)) <= v13 )
    return 0;
  v16 = sub_C336A0(*(_QWORD *)(v2 + 24)) - 1;
  v17 = *(_QWORD *)(v3 + 16);
  if ( !*(_BYTE *)(v17 + 4) )
  {
    *(_DWORD *)v17 = v16;
    *(_BYTE *)(v17 + 4) = 1;
    v17 = *(_QWORD *)(v3 + 16);
  }
  LOBYTE(v17) = *(_DWORD *)v17 == v16;
  LOBYTE(v16) = v16 > 0;
  return v16 & (unsigned int)v17;
}
