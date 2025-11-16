// Function: sub_92C930
// Address: 0x92c930
//
__int64 __fastcall sub_92C930(__int64 *a1, __int64 a2, char a3, unsigned __int64 a4, _DWORD *a5)
{
  __int64 v7; // rax
  char v8; // bl
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int8 v11; // al
  __int64 v13; // [rsp+0h] [rbp-40h]

  v7 = sub_91A390(*(_QWORD *)(*a1 + 32) + 8LL, a4, 0, a4);
  v8 = *(_BYTE *)(a4 + 140);
  v9 = v7;
  if ( v8 == 12 )
  {
    v10 = a4;
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
      v8 = *(_BYTE *)(v10 + 140);
    }
    while ( v8 == 12 );
  }
  v13 = v9;
  v11 = sub_91B6F0(a4);
  return sub_92BD50(a1, a2, a3, v13, v11, v8 == 1, a5);
}
