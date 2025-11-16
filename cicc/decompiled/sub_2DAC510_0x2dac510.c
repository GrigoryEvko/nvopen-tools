// Function: sub_2DAC510
// Address: 0x2dac510
//
__int64 __fastcall sub_2DAC510(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  unsigned int v3; // r15d
  __int64 v5; // rcx
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // rdx
  int v10; // eax

  if ( !(_BYTE)qword_501CF08 )
    return 0;
  v2 = *(__int64 (**)())(**(_QWORD **)a1 + 1768LL);
  if ( v2 == sub_2D9F320 )
    return 0;
  if ( !(unsigned __int8)v2() )
    return 0;
  v8 = *(_QWORD *)(a2 + 80);
  if ( v8 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    v3 = 0;
    do
    {
      v9 = v8 - 24;
      if ( !v8 )
        v9 = 0;
      v10 = sub_2DA98B0(*(_QWORD *)a1, *(__int64 **)(a1 + 8), v9, v5, v6, v7);
      v8 = *(_QWORD *)(v8 + 8);
      v3 |= v10;
    }
    while ( a2 + 72 != v8 );
  }
  return v3;
}
