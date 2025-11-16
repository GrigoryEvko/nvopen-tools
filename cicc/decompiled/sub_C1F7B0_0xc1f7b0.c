// Function: sub_C1F7B0
// Address: 0xc1f7b0
//
__int64 *__fastcall sub_C1F7B0(__int64 *a1, unsigned __int64 *a2, __int64 *a3)
{
  _QWORD *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rdx
  char *(*v12)(); // rcx
  char *v13; // rax
  unsigned __int64 v15; // rax
  _QWORD v16[4]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v17; // [rsp+20h] [rbp-70h]
  _QWORD v18[4]; // [rsp+30h] [rbp-60h] BYREF
  int v19; // [rsp+50h] [rbp-40h]
  _QWORD *v20; // [rsp+58h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F83B68) )
  {
    v5 = (_QWORD *)*a2;
    *a2 = 0;
    v6 = a3[1];
    v7 = *a3;
    v17 = 261;
    v8 = v5[7];
    v9 = v5[5];
    v16[0] = v5[6];
    v16[1] = v8;
    v10 = *(_QWORD *)v6;
    v11 = 14;
    v12 = *(char *(**)())(**(_QWORD **)v6 + 16LL);
    v13 = "Unknown buffer";
    if ( v12 != sub_C1E8B0 )
      v13 = (char *)((__int64 (__fastcall *)(__int64, __int64, __int64))v12)(v10, 261, 14);
    v18[2] = v13;
    v18[1] = 12;
    v20 = v16;
    v18[0] = &unk_49D9C78;
    v18[3] = v11;
    v19 = v9;
    sub_B6EB20(v7, (__int64)v18);
    *a1 = 1;
    (*(void (__fastcall **)(_QWORD *))(*v5 + 8LL))(v5);
  }
  else
  {
    v15 = *a2;
    *a2 = 0;
    *a1 = v15 | 1;
  }
  return a1;
}
