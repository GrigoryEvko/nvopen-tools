// Function: sub_39F8AF0
// Address: 0x39f8af0
//
void __fastcall sub_39F8AF0(__int64 a1, __int64 (__fastcall *a2)(__int64, _QWORD, _QWORD), __int64 a3)
{
  __int64 v3; // r14
  int v5; // ebx
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // r8d
  unsigned __int64 v10; // [rsp+8h] [rbp-40h]

  v3 = a3 + 16;
  v10 = *(_QWORD *)(a3 + 8);
  v5 = (v10 >> 1) - 1;
  if ( v5 >= 0 )
  {
    do
      sub_39F8A30(a1, a2, v3, v5, v10);
    while ( v5-- != 0 );
  }
  v7 = (int)v10 - 1;
  if ( (int)v10 - 1 > 0 )
  {
    do
    {
      v8 = *(_QWORD *)(a3 + 16);
      v9 = v7;
      *(_QWORD *)(a3 + 16) = *(_QWORD *)(a3 + 8 * v7 + 16);
      *(_QWORD *)(a3 + 8 * v7-- + 16) = v8;
      sub_39F8A30(a1, a2, v3, 0, v9);
    }
    while ( (int)v7 > 0 );
  }
}
