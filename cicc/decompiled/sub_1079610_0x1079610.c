// Function: sub_1079610
// Address: 0x1079610
//
__int64 __fastcall sub_1079610(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rdi
  char *v5; // rax
  _QWORD *v6; // r12
  __int64 v7; // r12
  _BYTE *v8; // rax
  int v9; // r14d
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  _QWORD *v12; // r12
  _QWORD *v13; // r12
  __int64 result; // rax

  v4 = **(_QWORD **)(a1 + 104);
  v5 = *(char **)(v4 + 32);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(v4 + 24) )
  {
    sub_CB5D20(v4, a3);
  }
  else
  {
    *(_QWORD *)(v4 + 32) = v5 + 1;
    *v5 = a3;
  }
  v6 = **(_QWORD ***)(a1 + 104);
  *(_QWORD *)a2 = (*(__int64 (__fastcall **)(_QWORD *))(*v6 + 80LL))(v6) + v6[4] - v6[2];
  v7 = **(_QWORD **)(a1 + 104);
  v8 = *(_BYTE **)(v7 + 32);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
  {
    sub_CB5D20(**(_QWORD **)(a1 + 104), 128);
  }
  else
  {
    *(_QWORD *)(v7 + 32) = v8 + 1;
    *v8 = 0x80;
  }
  v9 = 3;
  do
  {
    v10 = *(_BYTE **)(v7 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v7 + 24) )
    {
      sub_CB5D20(v7, 128);
    }
    else
    {
      *(_QWORD *)(v7 + 32) = v10 + 1;
      *v10 = 0x80;
    }
    --v9;
  }
  while ( v9 );
  v11 = *(_BYTE **)(v7 + 32);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v7 + 24) )
  {
    sub_CB5D20(v7, 0);
  }
  else
  {
    *(_QWORD *)(v7 + 32) = v11 + 1;
    *v11 = 0;
  }
  v12 = **(_QWORD ***)(a1 + 104);
  *(_QWORD *)(a2 + 16) = (*(__int64 (__fastcall **)(_QWORD *))(*v12 + 80LL))(v12) + v12[4] - v12[2];
  v13 = **(_QWORD ***)(a1 + 104);
  *(_QWORD *)(a2 + 8) = (*(__int64 (__fastcall **)(_QWORD *))(*v13 + 80LL))(v13) + v13[4] - v13[2];
  result = *(unsigned int *)(a1 + 1088);
  *(_DWORD *)(a1 + 1088) = result + 1;
  *(_DWORD *)(a2 + 24) = result;
  return result;
}
