// Function: sub_3215A30
// Address: 0x3215a30
//
__int64 __fastcall sub_3215A30(unsigned __int64 *a1, _QWORD *a2, __int16 a3)
{
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 v8; // r15
  __int64 (__fastcall *v9)(__int64, __int64, _QWORD); // rbx
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v13; // r14
  __int64 (__fastcall *v14)(__int64, _QWORD, _QWORD); // r15
  unsigned __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // r15
  __int64 (__fastcall *v18)(_QWORD *, __int64, __int64, _QWORD, __int64); // rbx
  unsigned __int64 v19; // rax
  unsigned int v20; // eax
  int v21; // [rsp+Ah] [rbp-36h] BYREF
  __int16 v22; // [rsp+Eh] [rbp-32h]

  if ( (unsigned __int16)a3 > 0x14u )
  {
    if ( a3 == 21 )
      return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*a2 + 424LL))(
               a2,
               *(unsigned int *)(*a1 + 16),
               0,
               0);
    goto LABEL_11;
  }
  if ( (unsigned __int16)a3 > 0x10u )
  {
    v13 = a2[28];
    v14 = *(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v13 + 536LL);
    v15 = sub_31DF6E0((__int64)a2);
    v21 = v15;
    v22 = WORD2(v15);
    v16 = sub_32159A0((__int64)a1, (__int64)&v21, a3);
    return v14(v13, *(unsigned int *)(*a1 + 16), v16);
  }
  if ( a3 != 16 )
LABEL_11:
    BUG();
  v5 = sub_3215130(*a1);
  v6 = sub_3215100(*a1);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 16LL);
  if ( v7 == sub_3214A40 || (v17 = ((__int64 (__fastcall *)(unsigned __int64))v7)(v6)) == 0 )
  {
    v8 = a2[28];
    v9 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v8 + 536LL);
    v10 = sub_31DF6E0((__int64)a2);
    v21 = v10;
    v22 = WORD2(v10);
    v11 = sub_32159A0((__int64)a1, (__int64)&v21, 16);
    return v9(v8, v5, v11);
  }
  else
  {
    v18 = *(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int64))(*a2 + 432LL);
    v19 = sub_31DF6E0((__int64)a2);
    v21 = v19;
    v22 = WORD2(v19);
    v20 = sub_32159A0((__int64)a1, (__int64)&v21, 16);
    return v18(a2, v17, v5, v20, 1);
  }
}
