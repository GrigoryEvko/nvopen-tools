// Function: sub_852630
// Address: 0x852630
//
__int64 __fastcall sub_852630(int a1, __int64 a2, const char *a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  int v7; // r13d
  __int64 v9; // rbx
  size_t v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  char *v15; // rax
  bool v16; // zf
  __int64 result; // rax

  v7 = a5;
  v9 = sub_851C10(a1, a2, (__int64)a3, (__int64)a4, a5, a6);
  if ( a1 == 2 )
    *(_DWORD *)(v9 + 12) = a2;
  if ( a3 )
  {
    v10 = strlen(a3);
    v15 = (char *)sub_822B10(v10 + 1, a2, v11, v12, v13, v14);
    *(_QWORD *)(v9 + 24) = v15;
    strcpy(v15, a3);
  }
  v16 = qword_4F5FB70 == 0;
  *(_QWORD *)(v9 + 32) = *a4;
  *(_DWORD *)(v9 + 32) = v7;
  if ( v16 )
    qword_4F5FB70 = v9;
  result = qword_4F5FB68;
  if ( qword_4F5FB68 )
    *(_QWORD *)qword_4F5FB68 = v9;
  qword_4F5FB68 = v9;
  return result;
}
