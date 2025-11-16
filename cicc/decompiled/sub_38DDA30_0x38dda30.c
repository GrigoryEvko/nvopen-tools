// Function: sub_38DDA30
// Address: 0x38dda30
//
void (*__fastcall sub_38DDA30(_QWORD *a1))()
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  void (*v5)(); // rax
  void (*result)(); // rax
  __int64 v7; // rdi
  const char *v8; // [rsp+0h] [rbp-30h] BYREF
  char v9; // [rsp+10h] [rbp-20h]
  char v10; // [rsp+11h] [rbp-1Fh]

  v2 = a1[4];
  if ( v2 != a1[3] && !*(_QWORD *)(v2 - 72) || (v3 = a1[7], a1[6] != v3) && !*(_QWORD *)(*(_QWORD *)(v3 - 8) + 8LL) )
  {
    v7 = a1[1];
    v10 = 1;
    v9 = 3;
    v8 = "Unfinished frame!";
    return (void (*)())sub_38BE3D0(v7, 0, (__int64)&v8);
  }
  v4 = a1[2];
  if ( !v4 || (v5 = *(void (**)())(*(_QWORD *)v4 + 72LL), v5 == nullsub_1938) )
  {
    result = *(void (**)())(*a1 + 1048LL);
    if ( result == nullsub_1938 )
      return result;
    return (void (*)())((__int64 (__fastcall *)(_QWORD *))result)(a1);
  }
  v5();
  result = *(void (**)())(*a1 + 1048LL);
  if ( result != nullsub_1938 )
    return (void (*)())((__int64 (__fastcall *)(_QWORD *))result)(a1);
  return result;
}
