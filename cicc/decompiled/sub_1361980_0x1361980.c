// Function: sub_1361980
// Address: 0x1361980
//
__int64 __fastcall sub_1361980(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // rax

  v4 = *(__int64 **)(a2 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_18:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F9D764 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_18;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F9D764);
  v8 = sub_14CF090(v7, a3);
  v9 = *(__int64 **)(a2 + 8);
  v10 = v8;
  v11 = *v9;
  v12 = v9[1];
  if ( v11 == v12 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_17;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8);
  v14 = sub_1632FA0(*(_QWORD *)(a3 + 40));
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = v13 + 360;
  *(_QWORD *)(a1 + 32) = v10;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 1;
  *(_QWORD *)(a1 + 8) = v14;
  v15 = (_QWORD *)(a1 + 80);
  do
  {
    if ( v15 )
    {
      *v15 = -8;
      v15[1] = 0;
      v15[2] = 0;
      v15[3] = 0;
      v15[4] = 0;
      v15[5] = -8;
      v15[6] = 0;
      v15[7] = 0;
      v15[8] = 0;
      v15[9] = 0;
    }
    v15 += 11;
  }
  while ( v15 != (_QWORD *)(a1 + 784) );
  *(_QWORD *)(a1 + 784) = 0;
  *(_QWORD *)(a1 + 792) = a1 + 824;
  *(_QWORD *)(a1 + 800) = a1 + 824;
  *(_QWORD *)(a1 + 896) = a1 + 928;
  *(_QWORD *)(a1 + 904) = a1 + 928;
  *(_QWORD *)(a1 + 808) = 8;
  *(_DWORD *)(a1 + 816) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 912) = 16;
  *(_DWORD *)(a1 + 920) = 0;
  return a1;
}
