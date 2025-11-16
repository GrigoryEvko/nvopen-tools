// Function: sub_14A9430
// Address: 0x14a9430
//
__int64 __fastcall sub_14A9430(__int64 a1)
{
  unsigned int v2; // r13d
  int v3; // ebx
  __int64 v4; // rax
  char v5; // dl
  unsigned int v6; // edx
  __int64 v7; // rdi
  unsigned int v8; // r15d

  if ( (unsigned int)*(_QWORD *)(*(_QWORD *)a1 + 32LL) )
  {
    v2 = 0;
    v3 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = sub_15A0A60(a1, v2);
        if ( !v4 )
          return 0;
        v5 = *(_BYTE *)(v4 + 16);
        if ( v5 != 9 )
          break;
LABEL_11:
        if ( v3 == ++v2 )
          return 1;
      }
      if ( v5 != 13 )
        return 0;
      v6 = *(_DWORD *)(v4 + 32);
      v7 = *(_QWORD *)(v4 + 24);
      v8 = v6 - 1;
      if ( v6 <= 0x40 )
      {
        if ( v7 != 1LL << v8 )
          return 0;
        goto LABEL_11;
      }
      if ( (*(_QWORD *)(v7 + 8LL * (v8 >> 6)) & (1LL << v8)) == 0 || v8 != (unsigned int)sub_16A58A0(v4 + 24) )
        return 0;
      if ( v3 == ++v2 )
        return 1;
    }
  }
  return 1;
}
