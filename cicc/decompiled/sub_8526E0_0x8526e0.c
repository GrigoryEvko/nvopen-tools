// Function: sub_8526E0
// Address: 0x8526e0
//
__int64 __fastcall sub_8526E0(int a1, __int64 a2, __int64 a3, const char *a4, __int64 a5, __int64 a6)
{
  char v6; // r13
  __int64 v8; // rax
  __int64 v9; // rbx
  size_t v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  char *v15; // rax
  __int64 result; // rax

  v6 = a3;
  v8 = sub_851C10(a1, a2, a3, (__int64)a4, a5, a6);
  *(_DWORD *)(v8 + 12) = a2;
  v9 = v8;
  *(_BYTE *)(v8 + 16) = v6;
  if ( a4 )
  {
    v10 = strlen(a4);
    v15 = (char *)sub_822B10(v10 + 1, a2, v11, v12, v13, v14);
    *(_QWORD *)(v9 + 24) = v15;
    strcpy(v15, a4);
  }
  if ( !qword_4F5FB60 )
    qword_4F5FB60 = v9;
  result = qword_4F5FB58;
  if ( qword_4F5FB58 )
    *(_QWORD *)qword_4F5FB58 = v9;
  qword_4F5FB58 = v9;
  return result;
}
