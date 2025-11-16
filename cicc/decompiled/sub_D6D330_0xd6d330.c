// Function: sub_D6D330
// Address: 0xd6d330
//
unsigned __int64 __fastcall sub_D6D330(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 v7; // rcx
  char *v8; // r15
  unsigned __int64 v9; // r9
  int v10; // eax
  unsigned __int64 *v11; // rdi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // r15
  _QWORD *v16; // rbx
  unsigned __int64 v17; // r15
  char *v19; // r15
  _QWORD v20[2]; // [rsp+10h] [rbp-140h] BYREF
  unsigned __int64 v21; // [rsp+20h] [rbp-130h]
  _QWORD v22[2]; // [rsp+30h] [rbp-120h] BYREF
  __int64 v23; // [rsp+40h] [rbp-110h]
  unsigned __int64 v24; // [rsp+50h] [rbp-100h] BYREF
  __int64 v25; // [rsp+58h] [rbp-F8h]
  _BYTE v26[240]; // [rsp+60h] [rbp-F0h] BYREF

  if ( !a2 )
    return 0;
  v21 = a2;
  v20[0] = 6;
  v20[1] = 0;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)v20);
  v6 = *(_QWORD *)(a2 + 16);
  v24 = (unsigned __int64)v26;
  v25 = 0x800000000LL;
  if ( v6 )
  {
    do
    {
      v22[0] = 6;
      v13 = *(_QWORD *)(v6 + 24);
      v22[1] = 0;
      if ( v13 )
      {
        v23 = v13;
        if ( v13 != -8192 && v13 != -4096 )
          sub_BD73F0((__int64)v22);
      }
      else
      {
        v23 = 0;
      }
      v7 = (unsigned int)v25;
      v8 = (char *)v22;
      a2 = v24;
      v9 = (unsigned int)v25 + 1LL;
      v10 = v25;
      if ( v9 > HIDWORD(v25) )
      {
        if ( v24 > (unsigned __int64)v22 || (unsigned __int64)v22 >= v24 + 24LL * (unsigned int)v25 )
        {
          sub_D6D200((__int64)&v24, (unsigned int)v25 + 1LL, a3, (unsigned int)v25, a5, v9);
          v7 = (unsigned int)v25;
          a2 = v24;
          v10 = v25;
        }
        else
        {
          v19 = (char *)v22 - v24;
          sub_D6D200((__int64)&v24, (unsigned int)v25 + 1LL, a3, (unsigned int)v25, a5, v9);
          a2 = v24;
          v7 = (unsigned int)v25;
          v8 = &v19[v24];
          v10 = v25;
        }
      }
      v11 = (unsigned __int64 *)(a2 + 24 * v7);
      if ( v11 )
      {
        *v11 = 6;
        v12 = *((_QWORD *)v8 + 2);
        v11[1] = 0;
        v11[2] = v12;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
        {
          a2 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(v11, a2);
        }
        v10 = v25;
      }
      LODWORD(v25) = v10 + 1;
      LOBYTE(a2) = v23 != -4096;
      if ( ((v23 != 0) & (unsigned __int8)a2) != 0 && v23 != -8192 )
        sub_BD60C0(v22);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
    v14 = (_QWORD *)v24;
    v15 = (_QWORD *)(v24 + 24LL * (unsigned int)v25);
    if ( (_QWORD *)v24 != v15 )
    {
      do
      {
        a2 = v14[2];
        if ( *(_BYTE *)a2 == 28 )
          sub_D6D630(a1);
        v14 += 3;
      }
      while ( v14 != v15 );
      v16 = (_QWORD *)v24;
      v17 = v21;
      v14 = (_QWORD *)(v24 + 24LL * (unsigned int)v25);
      if ( (_QWORD *)v24 != v14 )
      {
        do
        {
          v14 -= 3;
          sub_D68D70(v14);
        }
        while ( v16 != v14 );
        v14 = (_QWORD *)v24;
      }
      goto LABEL_29;
    }
  }
  else
  {
    v14 = v26;
  }
  v17 = v21;
LABEL_29:
  if ( v14 != (_QWORD *)v26 )
    _libc_free(v14, a2);
  sub_D68D70(v20);
  return v17;
}
