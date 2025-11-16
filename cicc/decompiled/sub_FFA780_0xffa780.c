// Function: sub_FFA780
// Address: 0xffa780
//
__int64 __fastcall sub_FFA780(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r9
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v13 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v6 = (__int64 *)(sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8);
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F8FBC8, a3);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v9 = v8 + 8;
  v10 = (_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 1;
  do
  {
    if ( v10 )
      *v10 = -4096;
    v10 += 2;
  }
  while ( (_QWORD *)(a1 + 168) != v10 );
  v11 = a1 + 184;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 1;
  do
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = -4096;
      *(_DWORD *)(v11 + 8) = 0x7FFFFFFF;
    }
    v11 += 24;
  }
  while ( v11 != a1 + 280 );
  sub_FF9360((_QWORD *)a1, a3, v13, v6, v7, v9);
  return a1;
}
