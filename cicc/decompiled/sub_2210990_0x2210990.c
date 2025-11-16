// Function: sub_2210990
// Address: 0x2210990
//
__int64 __fastcall sub_2210990(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        char *a6,
        char *a7,
        char **a8)
{
  __int64 result; // rax
  char *v10; // rcx
  __int64 v11[2]; // [rsp+0h] [rbp-28h] BYREF
  char *v12[3]; // [rsp+10h] [rbp-18h] BYREF

  v11[0] = a3;
  v11[1] = a4;
  v12[0] = a6;
  v12[1] = a7;
  result = sub_2210820(v11, v12, 0x10FFFFu, 1, 0);
  v10 = v12[0];
  *a5 = v11[0];
  *a8 = v10;
  return result;
}
