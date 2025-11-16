// Function: sub_A83A20
// Address: 0xa83a20
//
__int64 __fastcall sub_A83A20(unsigned int **a1, _BYTE *a2, _BYTE *a3, _BYTE *a4, __int64 a5)
{
  int v5; // r15d
  int v6; // ebx
  __int64 (*v7)(void); // rax
  __int64 v8; // r12
  __int64 v10; // rax
  int v11; // r9d
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  char v17; // [rsp+20h] [rbp-60h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v5 = (int)a3;
  v6 = (int)a4;
  v7 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 104LL);
  if ( (char *)v7 != (char *)sub_948040 )
  {
    v8 = v7();
LABEL_6:
    if ( v8 )
      return v8;
    goto LABEL_8;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u && *a4 <= 0x15u )
  {
    v8 = sub_AD5A90(a2, a3, a4, 0);
    goto LABEL_6;
  }
LABEL_8:
  v18 = 257;
  v10 = sub_BD2C40(72, 3);
  v8 = v10;
  if ( v10 )
    sub_B4DFA0(v10, (_DWORD)a2, v5, v6, (unsigned int)&v17, v11, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v8,
    a5,
    a1[7],
    a1[8]);
  v12 = *a1;
  v13 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v13 )
  {
    do
    {
      v14 = *((_QWORD *)v12 + 1);
      v15 = *v12;
      v12 += 4;
      sub_B99FD0(v8, v15, v14);
    }
    while ( (unsigned int *)v13 != v12 );
  }
  return v8;
}
