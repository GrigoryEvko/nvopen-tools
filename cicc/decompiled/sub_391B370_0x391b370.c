// Function: sub_391B370
// Address: 0x391b370
//
__int64 __fastcall sub_391B370(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rdi
  char *v6; // rax
  _QWORD *v7; // rbx
  int v8; // r14d
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __int64 v12; // r15
  char v13; // si
  char v14; // al
  char *v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // rbx
  __int64 result; // rax

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(char **)(v5 + 24);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 16) )
  {
    sub_16E7DE0(v5, a3);
  }
  else
  {
    *(_QWORD *)(v5 + 24) = v6 + 1;
    *v6 = a3;
  }
  v7 = *(_QWORD **)(a1 + 8);
  v8 = 5;
  v9 = (*(__int64 (__fastcall **)(_QWORD *))(*v7 + 64LL))(v7);
  v10 = v7[3] - v7[1];
  v11 = 0xFFFFFFFFLL;
  *(_QWORD *)a2 = v9 + v10;
  v12 = *(_QWORD *)(a1 + 8);
  do
  {
    while ( 1 )
    {
      v13 = v11 & 0x7F;
      v14 = v11 & 0x7F | 0x80;
      v11 >>= 7;
      if ( v11 )
        v13 = v14;
      v15 = *(char **)(v12 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v12 + 16) )
        break;
      *(_QWORD *)(v12 + 24) = v15 + 1;
      *v15 = v13;
      if ( !--v8 )
        goto LABEL_9;
    }
    sub_16E7DE0(v12, v13);
    --v8;
  }
  while ( v8 );
LABEL_9:
  v16 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a2 + 16) = (*(__int64 (__fastcall **)(_QWORD *))(*v16 + 64LL))(v16) + v16[3] - v16[1];
  v17 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a2 + 8) = (*(__int64 (__fastcall **)(_QWORD *))(*v17 + 64LL))(v17) + v17[3] - v17[1];
  result = *(unsigned int *)(a1 + 976);
  *(_DWORD *)(a1 + 976) = result + 1;
  *(_DWORD *)(a2 + 24) = result;
  return result;
}
