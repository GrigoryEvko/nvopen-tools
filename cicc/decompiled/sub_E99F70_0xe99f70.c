// Function: sub_E99F70
// Address: 0xe99f70
//
void (*__fastcall sub_E99F70(_QWORD *a1, _QWORD *a2))()
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  void (*v6)(); // rax
  void (*result)(); // rax
  __int64 v8; // rdi
  const char *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+20h] [rbp-20h]
  char v11; // [rsp+21h] [rbp-1Fh]

  v3 = a1[4];
  if ( v3 != a1[3] && !*(_QWORD *)(v3 - 88) || (v4 = a1[11], a1[10] != v4) && !*(_QWORD *)(*(_QWORD *)(v4 - 8) + 8LL) )
  {
    v8 = a1[1];
    v11 = 1;
    v9 = "Unfinished frame!";
    v10 = 3;
    return (void (*)())sub_E66880(v8, a2, (__int64)&v9);
  }
  v5 = a1[2];
  if ( !v5 || (v6 = *(void (**)())(*(_QWORD *)v5 + 80LL), v6 == nullsub_341) )
  {
    result = *(void (**)())(*a1 + 1264LL);
    if ( result == nullsub_341 )
      return result;
    return (void (*)())((__int64 (__fastcall *)(_QWORD *))result)(a1);
  }
  v6();
  result = *(void (**)())(*a1 + 1264LL);
  if ( result != nullsub_341 )
    return (void (*)())((__int64 (__fastcall *)(_QWORD *))result)(a1);
  return result;
}
