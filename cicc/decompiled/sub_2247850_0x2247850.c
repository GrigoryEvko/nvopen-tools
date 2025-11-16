// Function: sub_2247850
// Address: 0x2247850
//
bool __fastcall sub_2247850(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  unsigned __int8 v4; // bp
  _QWORD *v5; // rdi
  unsigned __int8 v6; // al
  unsigned __int8 v7; // dl
  unsigned __int8 v9; // r12
  _DWORD *v10; // rax
  int v11; // eax
  int *v12; // rax
  int v13; // ecx
  int v14; // eax
  unsigned __int8 v15; // [rsp+0h] [rbp-28h]

  v3 = *(_QWORD **)a1;
  v4 = *(_DWORD *)(a1 + 8) == -1;
  if ( (v4 & (v3 != 0)) != 0 )
  {
    v9 = v4 & (v3 != 0);
    v10 = (_DWORD *)v3[2];
    v11 = (unsigned __int64)v10 >= v3[3] ? (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 72LL))(v3) : *v10;
    v4 = 0;
    if ( v11 == -1 )
    {
      *(_QWORD *)a1 = 0;
      v4 = v9;
    }
  }
  v5 = *(_QWORD **)a2;
  v6 = *(_DWORD *)(a2 + 8) == -1;
  v7 = v6 & (*(_QWORD *)a2 != 0);
  if ( v7 )
  {
    v12 = (int *)v5[2];
    if ( (unsigned __int64)v12 >= v5[3] )
    {
      v15 = v7;
      v14 = (*(__int64 (__fastcall **)(_QWORD *))(*v5 + 72LL))(v5);
      v7 = v15;
      v13 = v14;
    }
    else
    {
      v13 = *v12;
    }
    v6 = 0;
    if ( v13 == -1 )
    {
      *(_QWORD *)a2 = 0;
      v6 = v7;
    }
  }
  return v4 == v6;
}
