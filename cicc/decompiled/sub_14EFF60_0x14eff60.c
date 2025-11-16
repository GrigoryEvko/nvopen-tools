// Function: sub_14EFF60
// Address: 0x14eff60
//
bool __fastcall sub_14EFF60(__int64 a1, __int64 a2, unsigned int *a3, unsigned int a4, __int64 *a5)
{
  __int64 v6; // r8
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r8
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v16; // rax
  _QWORD *v17; // [rsp+8h] [rbp-28h]

  v6 = *a3;
  if ( (_DWORD)v6 == *(_DWORD *)(a2 + 8) )
    return 1;
  v8 = (unsigned int)(v6 + 1);
  v9 = *a3;
  *a3 = v8;
  v10 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v6);
  v11 = v10;
  if ( !*(_BYTE *)(a1 + 1656) )
  {
    if ( (unsigned int)v10 >= a4 )
      goto LABEL_4;
LABEL_11:
    v13 = a1 + 552;
    goto LABEL_8;
  }
  v11 = a4 - v10;
  if ( a4 >= (unsigned int)v10 )
    goto LABEL_11;
LABEL_4:
  if ( (_DWORD)v8 != *(_DWORD *)(a2 + 8) )
  {
    *a3 = v9 + 2;
    v12 = sub_14EFEB0((_QWORD *)a1, *(_QWORD *)(*(_QWORD *)a2 + 8 * v8));
    if ( v12 && *(_BYTE *)(v12 + 8) == 8 )
    {
      v17 = (_QWORD *)v12;
      v16 = sub_1521F50(a1 + 608, v11);
      v14 = sub_1628DA0(*v17, v16);
      goto LABEL_9;
    }
    v13 = a1 + 552;
LABEL_8:
    v14 = sub_1522F40(v13, v11);
LABEL_9:
    *a5 = v14;
    return v14 == 0;
  }
  return 1;
}
