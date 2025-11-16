// Function: sub_73E3D0
// Address: 0x73e3d0
//
__int64 __fastcall sub_73E3D0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // rcx

  result = dword_4D03F94;
  while ( 1 )
  {
    v9 = *(_QWORD *)(a2 + 88);
    if ( (_DWORD)result )
    {
      v10 = *(__int64 **)(v9 + 32);
      if ( v10 )
        v9 = *v10;
    }
    a2 = *(_QWORD *)(a2 + 96);
    if ( !a2 || *(_BYTE *)(a2 + 80) == 7 )
      break;
    if ( a3
      || dword_4F077C4 != 2
      || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 40) + 32LL) + 168LL) + 113LL) != 2 )
    {
      v7 = *(_QWORD *)(a2 + 88);
      if ( (_DWORD)result )
      {
        v8 = *(__int64 **)(v7 + 32);
        if ( v8 )
          v7 = *v8;
      }
      sub_73E2D0(a1, v7);
      a1 = *(_QWORD *)(a1 + 72);
      result = dword_4D03F94;
    }
  }
  return result;
}
