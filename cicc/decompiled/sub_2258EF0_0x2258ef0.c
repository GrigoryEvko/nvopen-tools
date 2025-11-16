// Function: sub_2258EF0
// Address: 0x2258ef0
//
__int64 __fastcall sub_2258EF0(__int64 *a1, __int64 a2, __int64 a3, const char *a4)
{
  const char *v6; // r12
  const char *v7; // rax
  __int64 v8; // r15
  _QWORD *v9; // rax
  unsigned __int64 v10; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r13d
  __int64 v15; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v16[16]; // [rsp+20h] [rbp-80h] BYREF

  v6 = "<unnamed>";
  if ( a4 )
    v6 = a4;
  v7 = (const char *)strlen(v6);
  sub_C7DA90(v16, a2, a3, v6, v7, 0);
  v8 = v16[0];
  sub_CCBAC0((__int64)v16, (__int64)sub_22579C0, (__int64)sub_2257950, 0, 0);
  v9 = (_QWORD *)sub_22077B0(0x28u);
  v10 = (unsigned __int64)v9;
  if ( v9 )
  {
    v9[2] = 0;
    v9[3] = 0;
    v9[4] = 0;
  }
  *v9 = sub_22585F0;
  v9[1] = a1 + 10;
  v16[3] = v9;
  sub_CD41B0(&v15, v8);
  if ( v15 )
  {
    sub_12BC8A0(a1, *(__int64 (**)())(v15 + 8), *(_QWORD *)(v15 + 16) - *(_QWORD *)(v15 + 8), a4, v11, v12);
    v13 = 0;
    if ( v15 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
  }
  else
  {
    v13 = 4;
  }
  j_j___libc_free_0(v10);
  if ( v8 )
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 8LL))(v8, 40);
  return v13;
}
