// Function: sub_921630
// Address: 0x921630
//
__int64 __fastcall sub_921630(unsigned int **a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v8; // r13
  unsigned int v9; // r12d
  unsigned int *v10; // rdi
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v12; // r13
  unsigned int *v14; // rdx
  unsigned int v15; // r12d
  unsigned int *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int v21; // [rsp+Ch] [rbp-64h]
  char v22[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v23; // [rsp+30h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 8);
  v21 = sub_BCB060(v8);
  v9 = 39 - ((a4 == 0) - 1);
  if ( v21 > (unsigned int)sub_BCB060(a3) )
    v9 = 38;
  if ( v8 == a3 )
    return a2;
  v10 = a1[10];
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v10 + 120LL);
  if ( v11 != sub_920130 )
  {
    v12 = v11((__int64)v10, v9, (_BYTE *)a2, a3);
    goto LABEL_8;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(v9) )
      v12 = sub_ADAB70(v9, a2, a3, 0);
    else
      v12 = sub_AA93C0(v9, a2, a3);
LABEL_8:
    if ( v12 )
      return v12;
  }
  v23 = 257;
  v12 = sub_B51D30(v9, a2, a3, v22, 0, 0);
  if ( (unsigned __int8)sub_920620(v12) )
  {
    v14 = a1[12];
    v15 = *((_DWORD *)a1 + 26);
    if ( v14 )
      sub_B99FD0(v12, 3, v14);
    sub_B45150(v12, v15);
  }
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v12,
    a5,
    a1[7],
    a1[8]);
  v16 = *a1;
  v17 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v17 )
  {
    do
    {
      v18 = *((_QWORD *)v16 + 1);
      v19 = *v16;
      v16 += 4;
      sub_B99FD0(v12, v19, v18);
    }
    while ( (unsigned int *)v17 != v16 );
  }
  return v12;
}
