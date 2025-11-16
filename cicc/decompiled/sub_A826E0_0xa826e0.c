// Function: sub_A826E0
// Address: 0xa826e0
//
__int64 __fastcall sub_A826E0(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, unsigned int *a6)
{
  _BYTE *v6; // r10
  unsigned int *v7; // r11
  unsigned int *v10; // rdi
  char v11; // r9
  unsigned int v12; // ebx
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, _BYTE *); // r12
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned int *v19; // r11
  unsigned int *v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v25; // rax
  unsigned int *v26; // [rsp+0h] [rbp-80h]
  unsigned int *v27; // [rsp+0h] [rbp-80h]
  unsigned int *v29; // [rsp+10h] [rbp-70h]
  char v31; // [rsp+1Ch] [rbp-64h]
  char v32[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  v6 = a3;
  v7 = a6;
  v31 = BYTE4(a4);
  if ( *((_BYTE *)a1 + 108) )
    return sub_B35400((_DWORD)a1, 108, (_DWORD)a2, (_DWORD)a3, a4, a5, (__int16)a6, 0);
  v10 = a1[10];
  v11 = BYTE4(a4);
  v12 = a4;
  v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v10 + 40LL);
  v14 = *((_DWORD *)a1 + 26);
  v15 = v14;
  if ( BYTE4(a4) )
    v15 = (unsigned int)a4;
  if ( v13 == sub_928A40 )
  {
    if ( *a2 > 0x15u || *a3 > 0x15u )
      goto LABEL_12;
    v26 = v7;
    if ( (unsigned __int8)sub_AC47B0(18) )
      v16 = sub_AD5570(18, a2, a3, 0, 0);
    else
      v16 = sub_AABE40(18, a2, a3);
    v6 = a3;
    v11 = v31;
    v7 = v26;
    v17 = v16;
  }
  else
  {
    v27 = v7;
    v25 = ((__int64 (__fastcall *)(unsigned int *, __int64, _BYTE *, _BYTE *, __int64))v13)(v10, 18, a2, a3, v15);
    v7 = v27;
    v11 = v31;
    v6 = a3;
    v17 = v25;
  }
  if ( v17 )
    return v17;
  v14 = *((_DWORD *)a1 + 26);
LABEL_12:
  if ( !v11 )
    v12 = v14;
  v29 = v7;
  v33 = 257;
  v18 = sub_B504D0(18, a2, v6, v32, 0, 0);
  v19 = v29;
  v17 = v18;
  if ( v29 || (v19 = a1[12]) != 0 )
    sub_B99FD0(v18, 3, v19);
  sub_B45150(v17, v12);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v17,
    a5,
    a1[7],
    a1[8]);
  v20 = *a1;
  v21 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v21 != v20 )
  {
    v22 = *((_QWORD *)v20 + 1);
    v23 = *v20;
    v20 += 4;
    sub_B99FD0(v17, v23, v22);
  }
  return v17;
}
