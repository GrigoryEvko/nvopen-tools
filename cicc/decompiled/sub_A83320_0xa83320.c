// Function: sub_A83320
// Address: 0xa83320
//
__int64 __fastcall sub_A83320(unsigned int **a1, _BYTE *a2, __int64 a3, __int64 a4, char a5)
{
  unsigned int *v8; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  int v18; // [rsp+10h] [rbp-60h] BYREF
  char v19; // [rsp+14h] [rbp-5Ch]
  __int16 v20; // [rsp+30h] [rbp-40h]

  if ( *((_BYTE *)a1 + 108) )
  {
    v19 = 0;
    return sub_B358C0((_DWORD)a1, 141, (_DWORD)a2, a3, v18, a4, 0, 0);
  }
  v8 = a1[10];
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v8 + 120LL);
  if ( v9 != sub_920130 )
  {
    v10 = v9((__int64)v8, 43u, a2, a3);
    goto LABEL_6;
  }
  if ( *a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(43) )
      v10 = sub_ADAB70(43, a2, a3, 0);
    else
      v10 = sub_AA93C0(43, a2, a3);
LABEL_6:
    if ( v10 )
      return v10;
  }
  v20 = 257;
  v11 = sub_BD2C40(72, unk_3F10A14);
  v10 = v11;
  if ( v11 )
    sub_B51830(v11, a2, a3, &v18, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v10,
    a4,
    a1[7],
    a1[8]);
  v12 = *a1;
  v13 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v13 != v12 )
  {
    v14 = *((_QWORD *)v12 + 1);
    v15 = *v12;
    v12 += 4;
    sub_B99FD0(v10, v15, v14);
  }
  if ( a5 )
    sub_B448D0(v10, 1);
  return v10;
}
