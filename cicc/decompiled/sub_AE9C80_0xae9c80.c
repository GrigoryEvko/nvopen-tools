// Function: sub_AE9C80
// Address: 0xae9c80
//
__int64 __fastcall sub_AE9C80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  _QWORD *v7; // rdx
  __int64 v9; // [rsp+8h] [rbp-18h]

  v5 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v6 = *(_QWORD *)(a3 + 32 * (2 - v5));
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_QWORD **)(v6 + 24);
    if ( *(_DWORD *)(v6 + 32) > 0x40u )
      v7 = (_QWORD *)*v7;
    LOBYTE(v9) = 0;
    sub_AE6F30(a1, a2, *(_QWORD *)(a3 - 32 * v5), 8LL * (_QWORD)v7, v9);
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
}
