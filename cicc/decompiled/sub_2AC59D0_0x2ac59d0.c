// Function: sub_2AC59D0
// Address: 0x2ac59d0
//
unsigned __int64 __fastcall sub_2AC59D0(__int64 a1, _BYTE *a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // ecx
  unsigned int v5; // edx
  _QWORD *v6; // rax
  _BYTE *v7; // r9
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 result; // rax
  int v11; // eax
  int v12; // r10d

  if ( *a2 <= 0x1Cu )
    return sub_2AC42A0(*(_QWORD *)a1, (__int64)a2);
  v2 = *(_DWORD *)(a1 + 152);
  v3 = *(_QWORD *)(a1 + 136);
  if ( !v2 )
    return sub_2AC42A0(*(_QWORD *)a1, (__int64)a2);
  v4 = v2 - 1;
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (_QWORD *)(v3 + 16LL * v5);
  v7 = (_BYTE *)*v6;
  if ( a2 != (_BYTE *)*v6 )
  {
    v11 = 1;
    while ( v7 != (_BYTE *)-4096LL )
    {
      v12 = v11 + 1;
      v5 = v4 & (v11 + v5);
      v6 = (_QWORD *)(v3 + 16LL * v5);
      v7 = (_BYTE *)*v6;
      if ( a2 == (_BYTE *)*v6 )
        goto LABEL_4;
      v11 = v12;
    }
    return sub_2AC42A0(*(_QWORD *)a1, (__int64)a2);
  }
LABEL_4:
  v8 = v6[1];
  if ( !v8 )
    return sub_2AC42A0(*(_QWORD *)a1, (__int64)a2);
  v9 = *(_QWORD *)(v8 + 16);
  result = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v9 & 4) != 0 )
    return **(_QWORD **)result;
  return result;
}
