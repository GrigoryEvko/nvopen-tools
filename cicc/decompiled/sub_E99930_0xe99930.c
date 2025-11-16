// Function: sub_E99930
// Address: 0xe99930
//
__int64 __fastcall sub_E99930(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD *v4; // r14
  __int64 v5; // rdi
  __int64 (*v6)(); // rdx
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // r13
  void (*v13)(); // rax
  __int64 v14; // rsi
  const char *v15; // [rsp+0h] [rbp-60h] BYREF
  char v16; // [rsp+20h] [rbp-40h]
  char v17; // [rsp+21h] [rbp-3Fh]

  result = sub_E99590((__int64)a1, a2);
  if ( result )
  {
    v4 = (_QWORD *)result;
    if ( *(_QWORD *)(result + 80) )
    {
      v5 = a1[1];
      v17 = 1;
      v16 = 3;
      v15 = "Not all chained regions terminated!";
      sub_E66880(v5, a2, (__int64)&v15);
    }
    v6 = *(__int64 (**)())(*a1 + 88LL);
    v7 = 1;
    if ( v6 != sub_E97650 )
      v7 = ((__int64 (__fastcall *)(_QWORD *))v6)(a1);
    v8 = v4[2] == 0;
    v4[1] = v7;
    if ( v8 )
      v4[2] = v7;
    v9 = a1[10];
    v10 = a1[14];
    v11 = *a1;
    v12 = (a1[11] - v9) >> 3;
    if ( v10 != v12 )
    {
      while ( 1 )
      {
        v13 = *(void (**)())(v11 + 24);
        if ( v13 == nullsub_340 )
        {
          if ( ++v10 == v12 )
            return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(v11 + 176))(a1, v4[7], 0);
        }
        else
        {
          v14 = *(_QWORD *)(v9 + 8 * v10++);
          ((void (__fastcall *)(_QWORD *, __int64))v13)(a1, v14);
          v11 = *a1;
          if ( v10 == v12 )
            return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(v11 + 176))(a1, v4[7], 0);
        }
        v9 = a1[10];
      }
    }
    return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(v11 + 176))(a1, v4[7], 0);
  }
  return result;
}
