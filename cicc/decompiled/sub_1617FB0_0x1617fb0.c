// Function: sub_1617FB0
// Address: 0x1617fb0
//
void __fastcall sub_1617FB0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rax
  int v4; // r12d
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  const char *v11; // rsi
  unsigned __int64 v12; // rax
  size_t v13; // r13
  __int64 v14; // rax

  if ( (_DWORD)a3 == 3 )
  {
    v6 = sub_16E8CB0(a1, a2, a3);
    v7 = sub_1263B40(v6, "Invalid operation: Trying to assign a ModulePass to a FunctionPassManager for pass: ");
    v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v10 = *(_BYTE **)(v7 + 24);
    v11 = (const char *)v8;
    v12 = *(_QWORD *)(v7 + 16);
    v13 = v9;
    if ( v9 > v12 - (unsigned __int64)v10 )
    {
      v14 = sub_16E7EE0(v7, v11, v9);
      v10 = *(_BYTE **)(v14 + 24);
      v7 = v14;
      v12 = *(_QWORD *)(v14 + 16);
    }
    else if ( v9 )
    {
      memcpy(v10, v11, v9);
      v12 = *(_QWORD *)(v7 + 16);
      v10 = (_BYTE *)(v13 + *(_QWORD *)(v7 + 24));
      *(_QWORD *)(v7 + 24) = v10;
    }
    if ( (unsigned __int64)v10 >= v12 )
    {
      sub_16E7DE0(v7, 10);
    }
    else
    {
      *(_QWORD *)(v7 + 24) = v10 + 1;
      *v10 = 10;
    }
  }
  else
  {
    v3 = a2[1];
    v4 = a3;
    if ( v3 != *a2 )
    {
      while ( 1 )
      {
        v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8));
        if ( v4 == v5 || v5 <= 1 )
          break;
        sub_160FB80((__int64)a2);
        v3 = a2[1];
        if ( *a2 == v3 )
          goto LABEL_8;
      }
      v3 = a2[1];
    }
LABEL_8:
    sub_1617B20(*(_QWORD *)(v3 - 8), a1, 1);
  }
}
