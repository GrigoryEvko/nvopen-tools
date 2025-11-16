// Function: sub_E7BDB0
// Address: 0xe7bdb0
//
__int64 __fastcall sub_E7BDB0(__int64 *a1, __int64 *a2)
{
  int v3; // eax
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // r15
  int v13; // ebx
  __int64 v14; // rax
  _QWORD *v15; // r12
  _QWORD *v16; // rbx
  _QWORD *v17; // r14
  _QWORD *v18; // rdi
  __int64 v19; // rdi
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int16 v22; // [rsp+Ch] [rbp-44h]
  unsigned __int16 v23; // [rsp+Eh] [rbp-42h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  int v25; // [rsp+18h] [rbp-38h]
  int v26; // [rsp+1Ch] [rbp-34h]

  v3 = *((_DWORD *)a1 + 16);
  v4 = a1[4];
  a1[4] = 0;
  v5 = a1[5];
  v6 = a1[6];
  a1[5] = 0;
  v7 = *a1;
  v8 = a1[1];
  v26 = v3;
  v9 = a1[9];
  v10 = a1[2];
  a1[6] = 0;
  v11 = a1[3];
  v24 = v9;
  LOWORD(v9) = *((_WORD *)a1 + 40);
  *a1 = *a2;
  v22 = v9;
  LODWORD(v9) = *((_DWORD *)a1 + 21);
  a1[1] = a2[1];
  v12 = a1[7];
  v25 = v9;
  v23 = *((_WORD *)a1 + 44);
  a1[2] = a2[2];
  a1[3] = a2[3];
  a1[4] = a2[4];
  a1[5] = a2[5];
  a1[6] = a2[6];
  v13 = *((_DWORD *)a2 + 14);
  a2[4] = 0;
  a2[5] = 0;
  a2[6] = 0;
  *((_DWORD *)a1 + 14) = v13;
  *((_DWORD *)a1 + 15) = *((_DWORD *)a2 + 15);
  *((_DWORD *)a1 + 16) = *((_DWORD *)a2 + 16);
  a1[9] = a2[9];
  *((_BYTE *)a1 + 80) = *((_BYTE *)a2 + 80);
  *((_BYTE *)a1 + 81) = *((_BYTE *)a2 + 81);
  *((_DWORD *)a1 + 21) = *((_DWORD *)a2 + 21);
  *((_BYTE *)a1 + 88) = *((_BYTE *)a2 + 88);
  *((_BYTE *)a1 + 89) = *((_BYTE *)a2 + 89);
  *a2 = v7;
  a2[1] = v8;
  v14 = a2[6];
  v15 = (_QWORD *)a2[4];
  v16 = (_QWORD *)a2[5];
  a2[2] = v10;
  a2[3] = v11;
  v21 = v14;
  a2[4] = v4;
  a2[5] = v5;
  a2[6] = v6;
  if ( v15 != v16 )
  {
    v17 = v15;
    do
    {
      v18 = (_QWORD *)v17[9];
      if ( v18 != v17 + 11 )
        j_j___libc_free_0(v18, v17[11] + 1LL);
      v19 = v17[6];
      if ( v19 )
        j_j___libc_free_0(v19, v17[8] - v19);
      v17 += 13;
    }
    while ( v16 != v17 );
  }
  if ( v15 )
    j_j___libc_free_0(v15, v21 - (_QWORD)v15);
  a2[7] = v12;
  *((_DWORD *)a2 + 16) = v26;
  a2[9] = v24;
  *((_WORD *)a2 + 40) = v22;
  *((_DWORD *)a2 + 21) = v25;
  *((_WORD *)a2 + 44) = v23;
  return v23;
}
