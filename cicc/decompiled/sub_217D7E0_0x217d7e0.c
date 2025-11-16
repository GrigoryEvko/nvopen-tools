// Function: sub_217D7E0
// Address: 0x217d7e0
//
__int64 __fastcall sub_217D7E0(int a1, __int64 a2, _DWORD *a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rax

  if ( a3 )
    a3[2] = 0;
  if ( a1 < 0 )
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8LL * (unsigned int)a1);
  if ( v7 && ((*(_BYTE *)(v7 + 3) & 0x10) != 0 || (v7 = *(_QWORD *)(v7 + 32)) != 0 && (*(_BYTE *)(v7 + 3) & 0x10) != 0) )
  {
    v8 = 0;
    do
    {
      v9 = *(_QWORD *)(v7 + 16);
      if ( a3 )
      {
        v10 = (unsigned int)a3[2];
        if ( (unsigned int)v10 >= a3[3] )
        {
          sub_16CD150((__int64)a3, a3 + 4, 0, 8, a5, a6);
          v10 = (unsigned int)a3[2];
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v9;
        ++a3[2];
      }
      else
      {
        if ( v8 )
          return 0;
        v8 = *(_QWORD *)(v7 + 16);
      }
      v7 = *(_QWORD *)(v7 + 32);
    }
    while ( v7 && (*(_BYTE *)(v7 + 3) & 0x10) != 0 );
  }
  else
  {
    v8 = 0;
  }
  if ( a3 )
  {
    if ( a3[2] == 1 )
      return **(_QWORD **)a3;
    else
      return 0;
  }
  return v8;
}
