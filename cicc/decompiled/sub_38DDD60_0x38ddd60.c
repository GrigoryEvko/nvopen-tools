// Function: sub_38DDD60
// Address: 0x38ddd60
//
void (*__fastcall sub_38DDD60(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4))()
{
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rax
  unsigned int *v9; // rax
  __int64 v10; // rdi
  unsigned int *v11; // r14
  __int64 v12; // r15
  char *v14; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+10h] [rbp-40h]
  char v16; // [rsp+11h] [rbp-3Fh]

  v6 = a1[1];
  v7 = sub_38CF310(a3, 0, v6, 0);
  v8 = sub_38CF310(a2, 0, a1[1], 0);
  v9 = (unsigned int *)sub_38CB1F0(17, v8, v7, v6, 0);
  v10 = a1[1];
  v11 = v9;
  if ( !*(_BYTE *)(*(_QWORD *)(v10 + 16) + 296LL) )
    return (void (*)())sub_38DDD30((__int64)a1, v9);
  v16 = 1;
  v14 = "set";
  v15 = 3;
  v12 = sub_38BF8E0(v10, (__int64)&v14, 1, 1);
  (*(void (__fastcall **)(__int64 *, __int64, unsigned int *))(*a1 + 240))(a1, v12, v11);
  return sub_38DDC80(a1, v12, a4, 0);
}
