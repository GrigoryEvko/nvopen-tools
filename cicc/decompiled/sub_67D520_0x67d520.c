// Function: sub_67D520
// Address: 0x67d520
//
__int64 __fastcall sub_67D520(unsigned int a1, unsigned __int8 a2, unsigned int *a3)
{
  __int64 v3; // r9
  __int64 v4; // r10
  __int64 result; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rdx

  v3 = *a3;
  v4 = *((unsigned __int16 *)a3 + 2);
  result = qword_4CFDEC0[(v4 + 1) * (v3 + 1) * a1 * ((unsigned __int64)a2 + 1) % 0x3D7];
  if ( !result )
    return 0;
  while ( *(_DWORD *)(result + 8) != a1
       || *(_BYTE *)(result + 12) != a2
       || *(_DWORD *)(result + 16) != (_DWORD)v3
       || *(_WORD *)(result + 20) != (_WORD)v4 )
  {
    result = *(_QWORD *)result;
    if ( !result )
      return result;
  }
  v6 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_DWORD *)(result + 24) != *(_DWORD *)v6 || *(_BYTE *)(v6 + 4) == 9 )
  {
    *(_DWORD *)(result + 24) = *(_DWORD *)v6;
    *(_DWORD *)(result + 28) = 0;
    return 1;
  }
  else
  {
    v7 = (unsigned int)(*(_DWORD *)(result + 28) + 1);
    *(_DWORD *)(result + 28) = v7;
    return v7 <= unk_4F07478;
  }
}
