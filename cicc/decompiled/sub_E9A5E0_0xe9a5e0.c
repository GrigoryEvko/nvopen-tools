// Function: sub_E9A5E0
// Address: 0xe9a5e0
//
void (*__fastcall sub_E9A5E0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4))()
{
  _QWORD *v6; // rbx
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rax
  unsigned __int8 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdi
  unsigned __int8 *v13; // r14
  __int64 v14; // r15
  char *v16; // [rsp+0h] [rbp-60h] BYREF
  char v17; // [rsp+20h] [rbp-40h]
  char v18; // [rsp+21h] [rbp-3Fh]

  v6 = (_QWORD *)a1[1];
  v7 = sub_E808D0(a3, 0, v6, 0);
  v8 = sub_E808D0(a2, 0, (_QWORD *)a1[1], 0);
  v9 = (unsigned __int8 *)sub_E81A00(18, v8, v7, v6, 0);
  v12 = a1[1];
  v13 = v9;
  if ( !*(_BYTE *)(*(_QWORD *)(v12 + 152) + 280LL) )
    return (void (*)())sub_E9A5B0((__int64)a1, v9);
  v18 = 1;
  v16 = "set";
  v17 = 3;
  v14 = sub_E6C380(v12, (__int64 *)&v16, 1, v10, v11);
  (*(void (__fastcall **)(_QWORD *, __int64, unsigned __int8 *))(*a1 + 272LL))(a1, v14, v13);
  return sub_E9A500((__int64)a1, v14, a4, 0);
}
