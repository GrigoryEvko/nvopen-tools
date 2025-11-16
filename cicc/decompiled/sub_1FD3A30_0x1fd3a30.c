// Function: sub_1FD3A30
// Address: 0x1fd3a30
//
__int64 __fastcall sub_1FD3A30(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 144);
  v2 = *(_QWORD *)(a1 + 40);
  if ( v1 )
  {
    *(_QWORD *)(v2 + 792) = v1;
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 784LL) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 792LL) + 24LL);
    v3 = *(_QWORD *)(a1 + 40);
    v4 = *(_QWORD *)(v3 + 792);
    if ( !v4 )
      BUG();
    if ( (*(_BYTE *)v4 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
        v4 = *(_QWORD *)(v4 + 8);
    }
    *(_QWORD *)(v3 + 792) = *(_QWORD *)(v4 + 8);
  }
  else
  {
    *(_QWORD *)(v2 + 792) = sub_1DD5D10(*(_QWORD *)(v2 + 784));
  }
  v5 = *(_QWORD *)(a1 + 40);
  for ( result = *(_QWORD *)(v5 + 792); result != *(_QWORD *)(v5 + 784) + 24LL; result = *(_QWORD *)(v5 + 792) )
  {
    if ( **(_WORD **)(result + 16) != 3 )
      break;
    if ( (*(_BYTE *)result & 4) == 0 )
    {
      while ( (*(_BYTE *)(result + 46) & 8) != 0 )
        result = *(_QWORD *)(result + 8);
    }
    *(_QWORD *)(v5 + 792) = *(_QWORD *)(result + 8);
    v5 = *(_QWORD *)(a1 + 40);
  }
  return result;
}
