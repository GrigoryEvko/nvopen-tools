// Function: sub_67E8D0
// Address: 0x67e8d0
//
__int64 __fastcall sub_67E8D0(unsigned int a1, _DWORD *a2, __int64 a3, int a4, _QWORD *a5)
{
  __int64 v7; // rbx
  _DWORD *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx

  v7 = a4;
  v8 = sub_67D610(a1, a2, 2u);
  v9 = (unsigned int)dword_4D03A00;
  v10 = (__int64)v8;
  v11 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
  {
    v11 = sub_823020((unsigned int)dword_4D03A00, 40);
    v9 = (unsigned int)dword_4D03A00;
  }
  else
  {
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  }
  *(_QWORD *)(v11 + 8) = 0;
  *(_DWORD *)v11 = 3;
  *(_QWORD *)(v11 + 16) = a3;
  if ( !*(_QWORD *)(v10 + 184) )
    *(_QWORD *)(v10 + 184) = v11;
  v12 = *(_QWORD *)(v10 + 192);
  if ( v12 )
    *(_QWORD *)(v12 + 8) = v11;
  *(_QWORD *)(v10 + 192) = v11;
  v13 = qword_4D039F0;
  if ( !qword_4D039F0 || (_DWORD)v9 == -1 )
    v13 = sub_823020(v9, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v13 + 8) = 0;
  *(_DWORD *)v13 = 0;
  *(_QWORD *)(v13 + 16) = v7;
  if ( !*(_QWORD *)(v10 + 184) )
    *(_QWORD *)(v10 + 184) = v13;
  v14 = *(_QWORD *)(v10 + 192);
  if ( v14 )
    *(_QWORD *)(v14 + 8) = v13;
  *(_QWORD *)(v10 + 192) = v13;
  return sub_67C730(a5, v10);
}
