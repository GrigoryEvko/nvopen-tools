// Function: sub_A7C060
// Address: 0xa7c060
//
__int64 __fastcall sub_A7C060(unsigned int **a1, _BYTE *a2, unsigned int a3, char a4, __int64 a5, unsigned int *a6)
{
  char v6; // r11
  _BYTE *v7; // r10
  unsigned int v9; // r14d
  unsigned int *v12; // rdi
  bool v13; // zf
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *); // r8
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // rax
  unsigned int *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  char v26; // [rsp+7h] [rbp-69h]
  char v27; // [rsp+7h] [rbp-69h]
  char v28[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v29; // [rsp+30h] [rbp-40h]

  v6 = a4;
  v7 = a2;
  v9 = a3;
  v12 = a1[10];
  v13 = a4 == 0;
  v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*(_QWORD *)v12 + 48LL);
  v15 = *((_DWORD *)a1 + 26);
  v16 = v15;
  if ( !v13 )
    v16 = a3;
  if ( v14 == sub_9288C0 )
  {
    if ( *a2 > 0x15u )
      goto LABEL_9;
    v26 = v6;
    v17 = sub_AAAFF0(12, a2);
    v7 = a2;
    v6 = v26;
    v18 = v17;
  }
  else
  {
    v27 = v6;
    v25 = ((__int64 (__fastcall *)(unsigned int *, __int64, _BYTE *, __int64))v14)(v12, 12, a2, v16);
    v6 = v27;
    v7 = a2;
    v18 = v25;
  }
  if ( v18 )
    return v18;
  v15 = *((_DWORD *)a1 + 26);
LABEL_9:
  if ( !v6 )
    v9 = v15;
  v29 = 257;
  v20 = sub_B50340(12, v7, v28, 0, 0);
  v18 = v20;
  if ( a6 || (a6 = a1[12]) != 0 )
    sub_B99FD0(v20, 3, a6);
  sub_B45150(v18, v9);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v18,
    a5,
    a1[7],
    a1[8]);
  v21 = *a1;
  v22 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v22 != v21 )
  {
    v23 = *((_QWORD *)v21 + 1);
    v24 = *v21;
    v21 += 4;
    sub_B99FD0(v18, v24, v23);
  }
  return v18;
}
