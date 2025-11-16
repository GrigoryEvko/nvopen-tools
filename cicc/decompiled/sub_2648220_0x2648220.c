// Function: sub_2648220
// Address: 0x2648220
//
__int64 __fastcall sub_2648220(_QWORD *a1, char a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  _QWORD *v9; // r12
  char *v10; // rsi
  __int64 result; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // r15
  unsigned __int64 v14; // r13
  volatile signed __int32 *v15; // rdi
  unsigned __int64 v16; // rdi
  __int64 v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v8 = sub_22077B0(0x80u);
  v9 = (_QWORD *)v8;
  if ( v8 )
  {
    *(_BYTE *)v8 = a2;
    *(_WORD *)(v8 + 1) = 0;
    *(_QWORD *)(v8 + 8) = a4;
    *(_DWORD *)(v8 + 16) = a5;
    *(_QWORD *)(v8 + 24) = v8 + 40;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = 0;
    *(_QWORD *)(v8 + 64) = 0;
    *(_QWORD *)(v8 + 72) = 0;
    *(_QWORD *)(v8 + 80) = 0;
    *(_QWORD *)(v8 + 88) = 0;
    *(_QWORD *)(v8 + 96) = 0;
    *(_QWORD *)(v8 + 104) = 0;
    *(_QWORD *)(v8 + 112) = 0;
    *(_QWORD *)(v8 + 120) = 0;
  }
  v18[0] = v8;
  v10 = (char *)a1[41];
  if ( v10 == (char *)a1[42] )
  {
    sub_2647A60(a1 + 40, v10, v18);
    v9 = (_QWORD *)v18[0];
  }
  else
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = v8;
      a1[41] += 8LL;
      goto LABEL_6;
    }
    a1[41] = 8;
  }
  if ( v9 )
  {
    v12 = v9[12];
    if ( v12 )
      j_j___libc_free_0(v12);
    v13 = v9[10];
    v14 = v9[9];
    if ( v13 != v14 )
    {
      do
      {
        v15 = *(volatile signed __int32 **)(v14 + 8);
        if ( v15 )
          sub_A191D0(v15);
        v14 += 16LL;
      }
      while ( v13 != v14 );
      v14 = v9[9];
    }
    if ( v14 )
      j_j___libc_free_0(v14);
    sub_2644030(v9 + 6);
    v16 = v9[3];
    if ( (_QWORD *)v16 != v9 + 5 )
      _libc_free(v16);
    j_j___libc_free_0((unsigned __int64)v9);
  }
LABEL_6:
  result = *(_QWORD *)(a1[41] - 8LL);
  v18[0] = result;
  if ( a3 )
  {
    *(_QWORD *)sub_263EE40(a1 + 6, (unsigned __int64 *)v18) = a3;
    return v18[0];
  }
  return result;
}
