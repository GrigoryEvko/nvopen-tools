// Function: sub_15A0E50
// Address: 0x15a0e50
//
bool __fastcall sub_15A0E50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  unsigned int v6; // r14d
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r12
  __int64 v12; // r12

  if ( *((_BYTE *)a1 + 16) == 14 )
  {
    if ( a1[4] == sub_16982C0(a1, a2, a3, a4) )
      v4 = a1[5] + 8LL;
    else
      v4 = (__int64)(a1 + 4);
    return (*(_BYTE *)(v4 + 18) & 7) == 1;
  }
  else
  {
    if ( *(_BYTE *)(*a1 + 8LL) == 16 )
    {
      v6 = 0;
      v7 = *(_QWORD *)(*a1 + 32LL);
      if ( !v7 )
        return 1;
      while ( 1 )
      {
        v8 = sub_15A0A60((__int64)a1, v6);
        v11 = v8;
        if ( !v8 || *(_BYTE *)(v8 + 16) != 14 )
          break;
        v12 = *(_QWORD *)(v8 + 32) == sub_16982C0(a1, v6, v9, v10) ? *(_QWORD *)(v11 + 40) + 8LL : v11 + 32;
        if ( (*(_BYTE *)(v12 + 18) & 7) != 1 )
          break;
        if ( ++v6 == v7 )
          return 1;
      }
    }
    return 0;
  }
}
