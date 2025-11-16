// Function: sub_2EC9F20
// Address: 0x2ec9f20
//
void __fastcall sub_2EC9F20(__int64 a1, __int64 a2, __int64 *a3, char a4, int a5, int a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v10; // r8
  int v11; // r10d
  __int64 v12; // rsi
  __int64 v13; // r9
  unsigned __int64 v14; // rdx

  *(_QWORD *)(a2 + 16) = a3;
  *(_BYTE *)(a2 + 25) = a4;
  v6 = *(_QWORD *)(a1 + 136);
  if ( *(_BYTE *)(v6 + 4016) )
  {
    v7 = *(_QWORD *)(v6 + 4024);
    v10 = *(_QWORD *)(v6 + 4896);
    v11 = a2 + 26;
    v12 = *a3;
    v13 = (*(_QWORD *)(v6 + 4904) - v10) >> 2;
    if ( a4 )
    {
      sub_2F79480(a6, v12, v11, v10, v13, v13, v7);
    }
    else
    {
      v14 = *(_QWORD *)(v6 + 4000) + ((unsigned __int64)*((unsigned int *)a3 + 50) << 6);
      if ( LOBYTE(qword_50216E8[8]) )
        sub_2F778F0(a6, v12, v14, v11, v10, v13, v7);
      else
        sub_2F77B10(a5, v12, v14, v11, v10, v13, v7);
    }
  }
}
