// Function: sub_15AB430
// Address: 0x15ab430
//
__int64 __fastcall sub_15AB430(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  unsigned int v5; // edx
  __int64 result; // rax
  __int64 v7; // rax
  _QWORD *v8; // rsi
  unsigned int v9; // edi
  _QWORD *v10; // rcx

  v4 = *(_QWORD **)(a1 + 408);
  if ( *(_QWORD **)(a1 + 416) != v4 )
  {
LABEL_2:
    sub_16CCBA0(a1 + 400, a2);
    result = v5;
    if ( !(_BYTE)v5 )
      return result;
    goto LABEL_6;
  }
  v8 = &v4[*(unsigned int *)(a1 + 428)];
  v9 = *(_DWORD *)(a1 + 428);
  if ( v4 == v8 )
  {
LABEL_16:
    if ( v9 >= *(_DWORD *)(a1 + 424) )
      goto LABEL_2;
    *(_DWORD *)(a1 + 428) = v9 + 1;
    *v8 = a2;
    ++*(_QWORD *)(a1 + 400);
  }
  else
  {
    v10 = 0;
    do
    {
      if ( *v4 == a2 )
        return 0;
      if ( *v4 == -2 )
        v10 = v4;
      ++v4;
    }
    while ( v8 != v4 );
    if ( !v10 )
      goto LABEL_16;
    *v10 = a2;
    --*(_DWORD *)(a1 + 432);
    ++*(_QWORD *)(a1 + 400);
  }
LABEL_6:
  v7 = *(unsigned int *)(a1 + 168);
  if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 172) )
  {
    sub_16CD150(a1 + 160, a1 + 176, 0, 8);
    v7 = *(unsigned int *)(a1 + 168);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v7) = a2;
  ++*(_DWORD *)(a1 + 168);
  return 1;
}
