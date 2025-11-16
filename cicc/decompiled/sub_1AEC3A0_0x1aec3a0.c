// Function: sub_1AEC3A0
// Address: 0x1aec3a0
//
__int64 __fastcall sub_1AEC3A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  _QWORD *v3; // r12
  __int64 v4; // r14
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // rax

  v2 = 0;
  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(_QWORD *)(a1 + 40);
  while ( v3 )
  {
    v5 = v3;
    v3 = (_QWORD *)v3[1];
    if ( v4 != sub_1648700((__int64)v5)[5] )
    {
      if ( *v5 )
      {
        v6 = v5[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v6 = v3;
        if ( v3 )
          v3[2] = v3[2] & 3LL | v6;
      }
      *v5 = a2;
      if ( a2 )
      {
        v7 = *(_QWORD *)(a2 + 8);
        v5[1] = v7;
        if ( v7 )
          *(_QWORD *)(v7 + 16) = (unsigned __int64)(v5 + 1) | *(_QWORD *)(v7 + 16) & 3LL;
        v5[2] = (a2 + 8) | v5[2] & 3LL;
        *(_QWORD *)(a2 + 8) = v5;
      }
      ++v2;
    }
  }
  return v2;
}
