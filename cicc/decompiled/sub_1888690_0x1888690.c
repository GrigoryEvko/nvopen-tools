// Function: sub_1888690
// Address: 0x1888690
//
__int64 __fastcall sub_1888690(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  result = *(_QWORD *)(a1 + 16);
  v3 = *(_QWORD *)(a2 + 16);
  if ( result )
  {
    v4 = a2 + 8;
    v5 = a1 + 8;
    if ( v3 )
    {
      *(_QWORD *)(a1 + 16) = v3;
      v6 = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(a2 + 16) = result;
      v7 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 24) = v6;
      v8 = *(_QWORD *)(a2 + 32);
      *(_QWORD *)(a2 + 24) = v7;
      v9 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = v8;
      *(_QWORD *)(a2 + 32) = v9;
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) = v5;
      *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) = v4;
      result = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(a2 + 40) = result;
    }
    else
    {
      *(_DWORD *)(a2 + 8) = *(_DWORD *)(a1 + 8);
      v10 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a2 + 16) = v10;
      *(_QWORD *)(a2 + 24) = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a2 + 32) = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(v10 + 8) = v4;
      result = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a2 + 40) = result;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = v5;
      *(_QWORD *)(a1 + 32) = v5;
      *(_QWORD *)(a1 + 40) = 0;
    }
  }
  else if ( v3 )
  {
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    v11 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 + 16) = v11;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(v11 + 8) = a1 + 8;
    *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 24) = a2 + 8;
    *(_QWORD *)(a2 + 32) = a2 + 8;
    *(_QWORD *)(a2 + 40) = 0;
    return a2 + 8;
  }
  return result;
}
