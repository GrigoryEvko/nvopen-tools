// Function: sub_392D6E0
// Address: 0x392d6e0
//
__int64 __fastcall sub_392D6E0(const void *a1, const void *a2)
{
  int v4; // r13d
  int v5; // eax
  int v6; // r13d
  int v7; // edx
  __int64 result; // rax
  bool v9; // cl
  bool v10; // dl
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // r14
  const void *v13; // rdi
  const void *v14; // rsi
  int v15; // edx
  const void *v16; // rdi
  const void *v17; // rsi
  size_t v18; // r12
  size_t v19; // rbx
  unsigned int v20; // eax

  v4 = sub_38E2700(*(_QWORD *)a1);
  v5 = sub_38E2700(*(_QWORD *)a2);
  if ( v5 != 3 && v4 == 3 )
    goto LABEL_3;
  v9 = v5 == 3;
  if ( v4 != 3 )
  {
    result = 0xFFFFFFFFLL;
    if ( v9 )
      return result;
  }
  if ( v4 == 3 && v9 )
  {
    result = 0xFFFFFFFFLL;
    if ( *((_DWORD *)a1 + 2) < *((_DWORD *)a2 + 2) )
      return result;
    goto LABEL_3;
  }
  v11 = *((_QWORD *)a1 + 3);
  v12 = *((_QWORD *)a2 + 3);
  v13 = (const void *)*((_QWORD *)a1 + 2);
  v14 = (const void *)*((_QWORD *)a2 + 2);
  if ( v12 < v11 )
  {
    if ( !v12 )
      goto LABEL_3;
    v15 = memcmp(v13, v14, *((_QWORD *)a2 + 3));
    if ( !v15 )
      goto LABEL_21;
  }
  else if ( !v11 || (v15 = memcmp(v13, v14, *((_QWORD *)a1 + 3))) == 0 )
  {
    if ( v12 == v11 )
      goto LABEL_3;
LABEL_21:
    result = 0xFFFFFFFFLL;
    if ( v12 > v11 )
      return result;
    goto LABEL_3;
  }
  result = 0xFFFFFFFFLL;
  if ( v15 < 0 )
    return result;
LABEL_3:
  v6 = sub_38E2700(*(_QWORD *)a2);
  v7 = sub_38E2700(*(_QWORD *)a1);
  if ( v7 != 3 )
  {
    result = 0;
    if ( v6 == 3 )
      return result;
  }
  v10 = v7 == 3;
  if ( v6 != 3 )
  {
    result = 1;
    if ( v10 )
      return result;
  }
  if ( v6 == 3 && v10 )
    return *((_DWORD *)a2 + 2) < *((_DWORD *)a1 + 2);
  v16 = (const void *)*((_QWORD *)a2 + 2);
  v17 = (const void *)*((_QWORD *)a1 + 2);
  v18 = *((_QWORD *)a2 + 3);
  v19 = *((_QWORD *)a1 + 3);
  if ( v19 < v18 )
  {
    result = 0;
    if ( !v19 )
      return result;
    v20 = memcmp(v16, v17, v19);
    if ( !v20 )
      return v19 > v18;
    return v20 >> 31;
  }
  if ( v18 )
  {
    v20 = memcmp(v16, v17, v18);
    if ( v20 )
      return v20 >> 31;
  }
  result = 0;
  if ( v19 != v18 )
    return v19 > v18;
  return result;
}
