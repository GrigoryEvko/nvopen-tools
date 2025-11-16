// Function: sub_2D00FF0
// Address: 0x2d00ff0
//
__int64 __fastcall sub_2D00FF0(unsigned int *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  int v4; // eax
  __int64 v5; // r14
  int v6; // eax
  int v7; // edx
  int v8; // ebx
  int v9; // eax

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( v3 && !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
  {
    v4 = *(_DWORD *)(v3 + 36);
    if ( (v4 == 8923 || v4 == 9250)
      && (v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL) == 14) )
    {
      v8 = sub_2D00C30(a1, *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      if ( v8 != (unsigned int)sub_2D00C30(a1, (_BYTE *)a2) )
      {
        v9 = sub_2D00C30(a1, (_BYTE *)v5);
        sub_2D00AD0(a1, a2, v9);
        return 1;
      }
    }
    else
    {
      v6 = sub_2D00C30(a1, (_BYTE *)a2);
      v7 = a1[1];
      if ( v6 != v7 )
      {
        sub_2D00AD0(a1, a2, v7);
        return 1;
      }
    }
  }
  return 0;
}
