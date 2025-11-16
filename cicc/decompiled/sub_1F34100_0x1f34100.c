// Function: sub_1F34100
// Address: 0x1f34100
//
__int64 __fastcall sub_1F34100(_QWORD *a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rax
  __int64 v3; // r12
  __int16 v4; // dx
  __int16 v5; // ax
  __int64 v6; // rbx
  int v7; // eax
  __int64 v8; // r8
  int v9; // ebx
  int v10; // eax
  __int64 v11; // r8
  int v13; // eax

  v1 = 0;
  if ( (unsigned int)((__int64)(a1[12] - a1[11]) >> 3) != 1 )
    return 0;
  if ( a1[9] != a1[8] )
  {
    v2 = sub_1DD6100((__int64)a1);
    v1 = 1;
    v3 = v2;
    if ( (_QWORD *)v2 != a1 + 3 )
    {
      v4 = *(_WORD *)(v2 + 46);
      v5 = v4 & 4;
      if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
      {
        v6 = *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL);
        LOBYTE(v6) = (unsigned __int8)v6 >> 7;
      }
      else
      {
        LOBYTE(v13) = sub_1E15D00(v3, 0x80u, 1);
        v4 = *(_WORD *)(v3 + 46);
        LODWORD(v6) = v13;
        v5 = v4 & 4;
      }
      if ( v5 || (v4 & 8) == 0 )
      {
        v8 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) >> 5) & 1LL;
      }
      else
      {
        LOBYTE(v7) = sub_1E15D00(v3, 0x20u, 1);
        v4 = *(_WORD *)(v3 + 46);
        LODWORD(v8) = v7;
        v5 = v4 & 4;
      }
      v9 = v8 & v6;
      if ( v5 || (v4 & 8) == 0 )
      {
        v11 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) >> 8) & 1LL;
      }
      else
      {
        LOBYTE(v10) = sub_1E15D00(v3, 0x100u, 1);
        LODWORD(v11) = v10;
      }
      return v9 & ((unsigned int)v11 ^ 1);
    }
  }
  return v1;
}
