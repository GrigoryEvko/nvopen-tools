// Function: sub_A83CB0
// Address: 0xa83cb0
//
__int64 __fastcall sub_A83CB0(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r11d
  int v7; // r15d
  int v8; // ebx
  __int64 (*v9)(void); // rax
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v13; // rax
  unsigned int *v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rax
  int v19; // [rsp+0h] [rbp-70h]
  int v20; // [rsp+0h] [rbp-70h]
  int v21; // [rsp+0h] [rbp-70h]
  char v23; // [rsp+10h] [rbp-60h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  v6 = a5;
  v7 = (int)a3;
  v8 = a4;
  v9 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 112LL);
  if ( (char *)v9 != (char *)sub_9B6630 )
  {
    v21 = a5;
    v18 = v9();
    v6 = v21;
    v11 = v18;
LABEL_5:
    if ( v11 )
      return v11;
    goto LABEL_7;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    v19 = a5;
    v10 = sub_AD5CE0(a2, a3, a4, a5, 0, a6);
    v6 = v19;
    v11 = v10;
    goto LABEL_5;
  }
LABEL_7:
  v20 = v6;
  v24 = 257;
  v13 = sub_BD2C40(112, unk_3F1FE60);
  v11 = v13;
  if ( v13 )
    sub_B4E9E0(v13, (_DWORD)a2, v7, v8, v20, (unsigned int)&v23, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a6,
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
