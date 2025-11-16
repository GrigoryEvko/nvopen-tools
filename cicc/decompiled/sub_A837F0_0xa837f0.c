// Function: sub_A837F0
// Address: 0xa837f0
//
__int64 __fastcall sub_A837F0(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  __int64 (*v6)(void); // rax
  __int64 v7; // r12
  __int64 v9; // rax
  unsigned int *v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rsi
  char v14[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v15; // [rsp+20h] [rbp-40h]

  v6 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 96LL);
  if ( (char *)v6 != (char *)sub_948070 )
  {
    v7 = v6();
LABEL_5:
    if ( v7 )
      return v7;
    goto LABEL_7;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    v7 = sub_AD5840(a2, a3, 0);
    goto LABEL_5;
  }
LABEL_7:
  v15 = 257;
  v9 = sub_BD2C40(72, 2);
  v7 = v9;
  if ( v9 )
    sub_B4DE80(v9, a2, a3, v14, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v7,
    a4,
    a1[7],
    a1[8]);
  v10 = *a1;
  v11 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v11 )
  {
    do
    {
      v12 = *((_QWORD *)v10 + 1);
      v13 = *v10;
      v10 += 4;
      sub_B99FD0(v7, v13, v12);
    }
    while ( (unsigned int *)v11 != v10 );
  }
  return v7;
}
