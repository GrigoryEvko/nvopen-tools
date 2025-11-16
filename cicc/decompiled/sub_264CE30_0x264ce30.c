// Function: sub_264CE30
// Address: 0x264ce30
//
void __fastcall sub_264CE30(_QWORD **a1, _QWORD *a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // r13
  int *v7; // rbx
  int *v8; // r12
  unsigned int *v9; // r14
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  char v12; // r13
  unsigned __int64 *v13; // rdi
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // r14
  __int64 v20; // r13
  __int64 v21; // rdi
  __int64 v22; // rcx
  volatile signed __int32 *v23; // rdi
  volatile signed __int32 *v24; // rdi
  volatile signed __int32 *v25; // rax
  _QWORD *v26; // [rsp+0h] [rbp-A0h]
  volatile signed __int32 *v27; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+18h] [rbp-88h]
  _QWORD *v30; // [rsp+18h] [rbp-88h]
  _QWORD *v31; // [rsp+20h] [rbp-80h] BYREF
  volatile signed __int32 *v32; // [rsp+28h] [rbp-78h]
  int *v33; // [rsp+30h] [rbp-70h]
  int *v34; // [rsp+38h] [rbp-68h]
  _QWORD v35[12]; // [rsp+40h] [rbp-60h] BYREF

  v4 = *(__int64 **)(a3 + 72);
  v5 = *(__int64 **)(a3 + 80);
  if ( v5 == v4 )
  {
LABEL_12:
    v31 = 0;
    v9 = (unsigned int *)*a1;
    v10 = (_QWORD *)sub_22077B0(0x48u);
    v11 = v10;
    if ( v10 )
    {
      v10[1] = 0x100000001LL;
      *v10 = off_49D3C50;
      v12 = *((_BYTE *)v9 + 16);
      memset(v35, 0, 32);
      sub_264A680((__int64)v35, (__int64)(v9 + 6));
      v11[2] = a3;
      v11[3] = a2;
      *((_BYTE *)v11 + 32) = v12;
      *((_BYTE *)v11 + 33) = 0;
      v11[5] = 0;
      v11[6] = 0;
      v11[7] = 0;
      *((_DWORD *)v11 + 16) = 0;
      sub_2649AA0((__int64)(v11 + 5), (__int64)v35);
      sub_2342640((__int64)v35);
    }
    v32 = (volatile signed __int32 *)v11;
    v31 = v11 + 2;
    sub_2647660((unsigned __int64 *)(a3 + 72), &v31);
    v13 = a2 + 6;
    if ( (_QWORD *)(*a1)[1] != a2 )
    {
      sub_2647660(v13, &v31);
      goto LABEL_16;
    }
    v14 = *a1[2];
    v30 = a1[2];
    v15 = (_QWORD *)a2[7];
    v16 = v14 - a2[6];
    if ( v15 == (_QWORD *)a2[8] )
    {
      sub_2647470(v13, (char *)v14, &v31);
LABEL_32:
      v14 = a2[6] + v16;
LABEL_33:
      *v30 = v14;
      *a1[2] += 16LL;
LABEL_16:
      if ( v32 )
        sub_A191D0(v32);
      return;
    }
    if ( (_QWORD *)v14 != v15 )
    {
      v26 = v31;
      v27 = v32;
      if ( v32 )
      {
        if ( &_pthread_key_create )
          _InterlockedAdd(v32 + 2, 1u);
        else
          ++*((_DWORD *)v32 + 2);
        v15 = (_QWORD *)a2[7];
      }
      if ( v15 )
      {
        v17 = *(v15 - 2);
        *(v15 - 2) = 0;
        *v15 = v17;
        v18 = *(v15 - 1);
        *(v15 - 1) = 0;
        v15[1] = v18;
        v15 = (_QWORD *)a2[7];
      }
      v19 = v15 - 2;
      a2[7] = v15 + 2;
      v20 = ((__int64)v15 - v14 - 16) >> 4;
      if ( (__int64)v15 - v14 - 16 > 0 )
      {
        do
        {
          v21 = *(v19 - 2);
          v22 = *(v19 - 1);
          v19 -= 2;
          v19[1] = 0;
          *v19 = 0;
          v19[2] = v21;
          v23 = (volatile signed __int32 *)v19[3];
          v19[3] = v22;
          if ( v23 )
            sub_A191D0(v23);
          --v20;
        }
        while ( v20 );
      }
      v24 = *(volatile signed __int32 **)(v14 + 8);
      *(_QWORD *)v14 = v26;
      *(_QWORD *)(v14 + 8) = v27;
      if ( v24 )
        sub_A191D0(v24);
      goto LABEL_32;
    }
    if ( v14 )
    {
      *(_QWORD *)v14 = v31;
      v25 = v32;
      *(_QWORD *)(v14 + 8) = v32;
      if ( v25 )
      {
        if ( &_pthread_key_create )
        {
          _InterlockedAdd(v25 + 2, 1u);
          v15 = (_QWORD *)a2[7];
          v14 = a2[6] + v16;
          goto LABEL_39;
        }
        ++*((_DWORD *)v25 + 2);
      }
      v15 = (_QWORD *)a2[7];
      v14 = a2[6] + v16;
    }
LABEL_39:
    a2[7] = v15 + 2;
    goto LABEL_33;
  }
  while ( 1 )
  {
    v6 = *v4;
    if ( *(_QWORD **)(*v4 + 8) == a2 )
      break;
    v4 += 2;
    if ( v5 == v4 )
      goto LABEL_12;
  }
  v29 = (*a1)[4] + 4LL * *((unsigned int *)*a1 + 12);
  sub_22B0690(&v31, *a1 + 3);
  v7 = v33;
  v8 = v34;
  while ( (int *)v29 != v7 )
  {
    sub_22B6470((__int64)v35, v6 + 24, v7);
    do
      ++v7;
    while ( v7 != v8 && (unsigned int)*v7 > 0xFFFFFFFD );
  }
  *(_BYTE *)(v6 + 16) |= *((_BYTE *)*a1 + 16);
}
