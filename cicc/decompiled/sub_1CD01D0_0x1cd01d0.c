// Function: sub_1CD01D0
// Address: 0x1cd01d0
//
__int64 __fastcall sub_1CD01D0(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // rbx
  int v3; // eax
  int v4; // edx
  __int64 v5; // r14
  int v6; // r15d
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // rdi

  v1 = 0;
  v2 = a1;
  v3 = *(unsigned __int16 *)(a1 + 24);
  if ( (unsigned __int16)(v3 - 1) <= 2u )
  {
    v4 = 1;
    do
    {
      v2 = *(_QWORD *)(v2 + 32);
      v1 = v4++;
      v3 = *(unsigned __int16 *)(v2 + 24);
    }
    while ( (unsigned __int16)(v3 - 1) <= 2u );
  }
  if ( (unsigned int)(v3 - 7) > 2 && (unsigned int)(v3 - 4) > 1 )
  {
    ++v1;
  }
  else
  {
    v5 = *(_QWORD *)(v2 + 40);
    v6 = v5;
    if ( (_DWORD)v5 )
    {
      v7 = 0;
      v8 = 8LL * (unsigned int)v5;
      do
      {
        v9 = *(_QWORD *)(*(_QWORD *)(v2 + 32) + v7);
        v7 += 8;
        v6 += sub_1CD01D0(v9);
      }
      while ( v8 != v7 );
      v1 += v6;
    }
  }
  return v1;
}
