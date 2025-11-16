// Function: sub_2233E50
// Address: 0x2233e50
//
bool __fastcall sub_2233E50(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  unsigned __int8 v4; // bp
  _QWORD *v5; // rdi
  char v6; // bl
  char v7; // r12
  unsigned __int8 v9; // r12

  v3 = *(_QWORD **)a1;
  v4 = *(_DWORD *)(a1 + 8) == -1;
  if ( (v4 & (v3 != 0)) != 0 )
  {
    v9 = v4 & (v3 != 0);
    v4 = 0;
    if ( v3[2] >= v3[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v3 + 72LL))(v3) == -1 )
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
    v6 = 0;
    if ( v5[2] >= v5[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v5 + 72LL))(v5) == -1 )
    {
      *(_QWORD *)a2 = 0;
      v6 = v7;
    }
  }
  return v6 == (char)v4;
}
