// Function: sub_94D3D0
// Address: 0x94d3d0
//
__int64 __fastcall sub_94D3D0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (*v7)(void); // rax
  __int64 v8; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rsi
  char v18[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  v7 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 80LL);
  if ( (char *)v7 != (char *)sub_92FAE0 )
  {
    v8 = v7();
LABEL_4:
    if ( v8 )
      return v8;
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v8 = sub_AAADB0(a2, a3, a4);
    goto LABEL_4;
  }
LABEL_6:
  v19 = 257;
  v8 = sub_BD2C40(104, unk_3F10A14);
  if ( v8 )
  {
    v10 = sub_B501B0(*(_QWORD *)(a2 + 8), a3, a4);
    sub_B44260(v8, v10, 64, 1, 0, 0);
    if ( *(_QWORD *)(v8 - 32) )
    {
      v11 = *(_QWORD *)(v8 - 24);
      **(_QWORD **)(v8 - 16) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v8 - 16);
    }
    *(_QWORD *)(v8 - 32) = a2;
    v12 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v8 - 24) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = v8 - 24;
    *(_QWORD *)(v8 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v8 - 32;
    *(_QWORD *)(v8 + 72) = v8 + 88;
    *(_QWORD *)(v8 + 80) = 0x400000000LL;
    sub_B50030(v8, a3, a4, v18);
  }
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v8,
    a5,
    a1[7],
    a1[8]);
  v13 = *a1;
  v14 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v14 )
  {
    do
    {
      v15 = *((_QWORD *)v13 + 1);
      v16 = *v13;
      v13 += 4;
      sub_B99FD0(v8, v16, v15);
    }
    while ( (unsigned int *)v14 != v13 );
  }
  return v8;
}
