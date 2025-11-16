// Function: sub_16484A0
// Address: 0x16484a0
//
__int64 __fastcall sub_16484A0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx

  result = *a1;
  v3 = *a2;
  if ( *a1 != *a2 )
  {
    if ( result )
    {
      v4 = a1[1];
      v5 = a1[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v5 = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = *(_QWORD *)(v4 + 16) & 3LL | v5;
      v3 = *a2;
      result = *a1;
    }
    if ( v3 )
    {
      v6 = a2[1];
      v7 = a2[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v7 = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
      v8 = *a2;
      *a1 = *a2;
      v9 = *(_QWORD *)(v8 + 8);
      a1[1] = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = (unsigned __int64)(a1 + 1) | *(_QWORD *)(v9 + 16) & 3LL;
      a1[2] = (v8 + 8) | a1[2] & 3;
      *(_QWORD *)(v8 + 8) = a1;
    }
    else
    {
      *a1 = 0;
    }
    if ( result )
    {
      *a2 = result;
      v10 = *(_QWORD *)(result + 8);
      a2[1] = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = (unsigned __int64)(a2 + 1) | *(_QWORD *)(v10 + 16) & 3LL;
      a2[2] = (result + 8) | a2[2] & 3;
      *(_QWORD *)(result + 8) = a2;
    }
    else
    {
      *a2 = 0;
    }
  }
  return result;
}
