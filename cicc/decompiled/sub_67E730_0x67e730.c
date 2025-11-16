// Function: sub_67E730
// Address: 0x67e730
//
__int64 __fastcall sub_67E730(unsigned int a1, _DWORD *a2, __int64 a3, int a4, __int64 a5, _QWORD *a6)
{
  __int64 v7; // r14
  _DWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx

  v7 = a4;
  v10 = sub_67D610(a1, a2, 2u);
  v11 = sub_67BB20(4);
  *(_QWORD *)(v11 + 16) = a3;
  *(_DWORD *)(v11 + 24) = -1;
  if ( !*((_QWORD *)v10 + 23) )
    *((_QWORD *)v10 + 23) = v11;
  v12 = *((_QWORD *)v10 + 24);
  if ( v12 )
    *(_QWORD *)(v12 + 8) = v11;
  *((_QWORD *)v10 + 24) = v11;
  v13 = qword_4D039F0;
  v14 = (unsigned int)dword_4D03A00;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
  {
    v13 = sub_823020((unsigned int)dword_4D03A00, 40);
    v14 = (unsigned int)dword_4D03A00;
  }
  else
  {
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  }
  *(_QWORD *)(v13 + 8) = 0;
  *(_DWORD *)v13 = 0;
  *(_QWORD *)(v13 + 16) = v7;
  if ( !*((_QWORD *)v10 + 23) )
    *((_QWORD *)v10 + 23) = v13;
  v15 = *((_QWORD *)v10 + 24);
  if ( v15 )
    *(_QWORD *)(v15 + 8) = v13;
  *((_QWORD *)v10 + 24) = v13;
  v16 = qword_4D039F0;
  if ( !qword_4D039F0 || (_DWORD)v14 == -1 )
    v16 = sub_823020(v14, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v16 + 8) = 0;
  *(_DWORD *)v16 = 5;
  *(_QWORD *)(v16 + 16) = a5;
  if ( !*((_QWORD *)v10 + 23) )
    *((_QWORD *)v10 + 23) = v16;
  v17 = *((_QWORD *)v10 + 24);
  if ( v17 )
    *(_QWORD *)(v17 + 8) = v16;
  *((_QWORD *)v10 + 24) = v16;
  return sub_67C730(a6, (__int64)v10);
}
