// Function: sub_177F910
// Address: 0x177f910
//
__int64 __fastcall sub_177F910(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int8 v5; // al
  unsigned int v6; // esi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v10; // rax
  unsigned int v11; // esi
  unsigned int v12; // r15d
  int v13; // r13d
  __int64 v14; // rax
  char v15; // dl
  unsigned int v16; // edx
  __int64 v17; // rax

  v5 = *((_BYTE *)a1 + 16);
  if ( v5 != 13 )
  {
    LOBYTE(v4) = v5 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
    if ( (_BYTE)v4 )
    {
      v10 = sub_15A1020(a1, a2, *(_QWORD *)a1, a4);
      if ( v10 && *(_BYTE *)(v10 + 16) == 13 )
      {
        v11 = *(_DWORD *)(v10 + 32);
        v8 = *(_QWORD *)(v10 + 24);
        v7 = 1LL << ((unsigned __int8)v11 - 1);
        if ( v11 > 0x40 )
          v8 = *(_QWORD *)(v8 + 8LL * ((v11 - 1) >> 6));
        goto LABEL_4;
      }
      v12 = 0;
      v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v13 )
        return v4;
      while ( 1 )
      {
        v14 = sub_15A0A60((__int64)a1, v12);
        if ( !v14 )
          break;
        v15 = *(_BYTE *)(v14 + 16);
        if ( v15 != 9 )
        {
          if ( v15 != 13 )
            break;
          v16 = *(_DWORD *)(v14 + 32);
          v17 = *(_QWORD *)(v14 + 24);
          if ( v16 > 0x40 )
            v17 = *(_QWORD *)(v17 + 8LL * ((v16 - 1) >> 6));
          if ( (v17 & (1LL << ((unsigned __int8)v16 - 1))) == 0 )
            break;
        }
        if ( v13 == ++v12 )
          return v4;
      }
    }
    return 0;
  }
  v6 = a1[8];
  v7 = *((_QWORD *)a1 + 3);
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 > 0x40 )
    v7 = *(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6));
LABEL_4:
  LOBYTE(v4) = (v8 & v7) != 0;
  return v4;
}
