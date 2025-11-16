// Function: sub_864420
// Address: 0x864420
//
__int64 __fastcall sub_864420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  int v6; // r9d
  unsigned int v7; // ebx
  __int64 v8; // rax
  unsigned int v9; // r13d
  char v10; // al
  bool v11; // r12
  __int64 v13; // rdi
  bool v14; // zf
  unsigned int v15; // eax
  unsigned int v16; // edx
  __int64 v17; // [rsp+8h] [rbp-38h]

  v5 = a1;
  v6 = a2;
  v7 = a3;
  v8 = *(_QWORD *)a1;
  v9 = dword_4F04C64;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 0x10) != 0 )
  {
    v15 = sub_864420(*(_QWORD *)(v8 + 64), a2, a3, a4, a5, (unsigned int)a2);
    v6 = a2;
    v16 = v15;
    v5 = a1;
    v11 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) & 0x10) != 0;
    v14 = dword_4F04C64 == v9;
    v9 = v15;
    v10 = v14;
    if ( v16 == -1 )
      v9 = dword_4F04C64;
  }
  else if ( (_DWORD)a5 || (v13 = *(_QWORD *)(v8 + 64)) == 0 )
  {
    v10 = 1;
    v11 = 0;
  }
  else
  {
    v17 = v5;
    if ( !(_DWORD)a3 )
    {
      sub_864360(v13, a4);
      v11 = 1;
      v9 = dword_4F04C64;
      v5 = v17;
      v6 = a2;
      goto LABEL_7;
    }
    sub_864230(v13, (unsigned int)a4);
    v11 = 1;
    v5 = v17;
    v6 = a2;
    v14 = v9 == dword_4F04C64;
    v9 = dword_4F04C64;
    v10 = v14;
  }
  if ( !v7 || (v7 = 0x40000, !v10) )
    v7 = 0;
LABEL_7:
  if ( v6 )
    v7 |= 0x400000u;
  sub_85C120(7u, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 168) + 152LL) + 24LL), v5, 0, 0, 0, 0, 0, 0, 0, 0, 0, v7);
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8)
                                                           & 0xEF
                                                           | (16 * v11);
  return v9;
}
