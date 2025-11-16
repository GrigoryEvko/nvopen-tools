// Function: sub_2553970
// Address: 0x2553970
//
void __fastcall sub_2553970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 8) )
    sub_2538630(a1, a2, a3, a4, a5, a6);
  v8 = a1 + 56;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 56;
  *(_QWORD *)(a1 + 80) = a1 + 56;
  *(_QWORD *)(a1 + 88) = 0;
  v9 = *(_QWORD *)(a2 + 64);
  if ( v9 )
  {
    v10 = sub_25383A0(v9, v8);
    v11 = v10;
    do
    {
      v12 = v10;
      v10 = *(_QWORD *)(v10 + 16);
    }
    while ( v10 );
    *(_QWORD *)(a1 + 72) = v12;
    v13 = v11;
    do
    {
      v14 = v13;
      v13 = *(_QWORD *)(v13 + 24);
    }
    while ( v13 );
    v15 = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(a1 + 80) = v14;
    *(_QWORD *)(a1 + 64) = v11;
    *(_QWORD *)(a1 + 88) = v15;
  }
}
