// Function: sub_1CBC100
// Address: 0x1cbc100
//
__int64 __fastcall sub_1CBC100(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax

  v4 = a1 + 8;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  *(_QWORD *)(a1 + 40) = 0;
  v5 = *(_QWORD *)(a2 + 24);
  if ( v5 )
  {
    v6 = sub_1CBA660(v5, v4);
    v7 = v6;
    do
    {
      v8 = v6;
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v6 );
    *(_QWORD *)(a1 + 24) = v8;
    v9 = v7;
    do
    {
      v10 = v9;
      v9 = *(_QWORD *)(v9 + 24);
    }
    while ( v9 );
    *(_QWORD *)(a1 + 32) = v10;
    v11 = *(_QWORD *)(a2 + 48);
    *(_QWORD *)(a1 + 16) = v7;
    *(_QWORD *)(a1 + 40) = v11;
  }
  return a1;
}
