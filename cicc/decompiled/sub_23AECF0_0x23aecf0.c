// Function: sub_23AECF0
// Address: 0x23aecf0
//
__int64 __fastcall sub_23AECF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // rcx
  _DWORD *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  bool v9; // zf
  char v11; // [rsp+Bh] [rbp-5h] BYREF
  int v12; // [rsp+Ch] [rbp-4h] BYREF

  v3 = a3;
  if ( a2 )
  {
    v4 = a2;
    if ( !a3 )
      v3 = *a1;
  }
  else
  {
    v4 = *a1;
  }
  v5 = (_DWORD *)a1[2];
  v6 = a1[1];
  v7 = (unsigned int)*v5;
  v8 = (unsigned int)(v7 + 1);
  *v5 = v8;
  v9 = *(_QWORD *)(v6 + 16) == 0;
  v11 = 1;
  v12 = v7;
  if ( v9 )
    sub_4263D6(v8, v5, v7);
  return (*(__int64 (__fastcall **)(__int64, char *, int *, __int64, __int64))(v6 + 24))(v6, &v11, &v12, v4, v3);
}
