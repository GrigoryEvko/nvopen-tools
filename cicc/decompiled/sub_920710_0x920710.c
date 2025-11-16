// Function: sub_920710
// Address: 0x920710
//
__int64 __fastcall sub_920710(
        unsigned int **a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        unsigned int a7,
        char a8)
{
  __int64 (*v10)(void); // rax
  __int64 v11; // r12
  unsigned int *v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v20[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  if ( *(_QWORD *)(a3 + 8) == a4 )
    return a3;
  v10 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 120LL);
  if ( (char *)v10 != (char *)sub_920130 )
  {
    v11 = v10();
    goto LABEL_6;
  }
  if ( *(_BYTE *)a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(a2) )
      v11 = sub_ADAB70(a2, a3, a4, 0);
    else
      v11 = sub_AA93C0(a2, a3, a4);
LABEL_6:
    if ( v11 )
      return v11;
  }
  v21 = 257;
  v11 = sub_B51D30(a2, a3, a4, v20, 0, 0);
  if ( (unsigned __int8)sub_920620(v11) )
  {
    if ( !a8 )
      a7 = *((_DWORD *)a1 + 26);
    if ( a6 || (a6 = a1[12]) != 0 )
      sub_B99FD0(v11, 3, a6);
    sub_B45150(v11, a7);
  }
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a5,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v15 )
  {
    do
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *v14;
      v14 += 4;
      sub_B99FD0(v11, v17, v16);
    }
    while ( (unsigned int *)v15 != v14 );
  }
  return v11;
}
