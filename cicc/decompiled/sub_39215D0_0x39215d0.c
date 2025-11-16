// Function: sub_39215D0
// Address: 0x39215d0
//
__int64 __fastcall sub_39215D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  unsigned __int64 v4; // rbx
  char v5; // si
  char v6; // al
  char *v7; // rax
  __int64 *v8; // rbx
  unsigned __int64 v9; // r12
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  char v16; // al
  char v17; // si
  char v18; // r12
  char *v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // r13
  unsigned __int64 v23; // r14
  char v24; // si
  char v25; // al
  char *v26; // rax
  _QWORD *v27; // r14
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 *v30; // [rsp-60h] [rbp-60h]
  _QWORD v31[2]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v32; // [rsp-48h] [rbp-48h]
  int v33; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 704);
  if ( (_DWORD)result )
  {
    sub_391B370(a1, (__int64)v31, 11);
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(unsigned int *)(a1 + 704);
    *(_DWORD *)(a1 + 88) = v33;
    do
    {
      while ( 1 )
      {
        v5 = v4 & 0x7F;
        v6 = v4 & 0x7F | 0x80;
        v4 >>= 7;
        if ( v4 )
          v5 = v6;
        v7 = *(char **)(v3 + 24);
        if ( (unsigned __int64)v7 >= *(_QWORD *)(v3 + 16) )
          break;
        *(_QWORD *)(v3 + 24) = v7 + 1;
        *v7 = v5;
        if ( !v4 )
          goto LABEL_8;
      }
      sub_16E7DE0(v3, v5);
    }
    while ( v4 );
LABEL_8:
    v8 = *(__int64 **)(a1 + 696);
    v9 = (unsigned __int64)*(unsigned int *)(a1 + 704) << 6;
    v30 = (__int64 *)((char *)v8 + v9);
    while ( v30 != v8 )
    {
      v10 = *(_QWORD *)(a1 + 8);
      v11 = *(_BYTE **)(v10 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
      {
        sub_16E7DE0(v10, 0);
        v12 = *(_QWORD *)(a1 + 8);
        v13 = *(_BYTE **)(v12 + 24);
        if ( (unsigned __int64)v13 < *(_QWORD *)(v12 + 16) )
        {
LABEL_11:
          *(_QWORD *)(v12 + 24) = v13 + 1;
          *v13 = 65;
          goto LABEL_12;
        }
      }
      else
      {
        *(_QWORD *)(v10 + 24) = v11 + 1;
        *v11 = 0;
        v12 = *(_QWORD *)(a1 + 8);
        v13 = *(_BYTE **)(v12 + 24);
        if ( (unsigned __int64)v13 < *(_QWORD *)(v12 + 16) )
          goto LABEL_11;
      }
      sub_16E7DE0(v12, 65);
LABEL_12:
      v14 = *(_QWORD *)(a1 + 8);
      v15 = *((unsigned int *)v8 + 6);
      do
      {
        while ( 1 )
        {
          v16 = v15;
          v17 = v15 & 0x7F;
          v15 >>= 7;
          if ( v15 || (v18 = 0, (v16 & 0x40) != 0) )
          {
            v17 |= 0x80u;
            v18 = 1;
          }
          v19 = *(char **)(v14 + 24);
          if ( (unsigned __int64)v19 >= *(_QWORD *)(v14 + 16) )
            break;
          *(_QWORD *)(v14 + 24) = v19 + 1;
          *v19 = v17;
          if ( !v18 )
            goto LABEL_19;
        }
        sub_16E7DE0(v14, v17);
      }
      while ( v18 );
LABEL_19:
      v20 = *(_QWORD *)(a1 + 8);
      v21 = *(_BYTE **)(v20 + 24);
      if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 16) )
      {
        sub_16E7DE0(v20, 11);
      }
      else
      {
        *(_QWORD *)(v20 + 24) = v21 + 1;
        *v21 = 11;
      }
      v22 = *(_QWORD *)(a1 + 8);
      v23 = *((unsigned int *)v8 + 12);
      do
      {
        while ( 1 )
        {
          v24 = v23 & 0x7F;
          v25 = v23 & 0x7F | 0x80;
          v23 >>= 7;
          if ( v23 )
            v24 = v25;
          v26 = *(char **)(v22 + 24);
          if ( (unsigned __int64)v26 >= *(_QWORD *)(v22 + 16) )
            break;
          *(_QWORD *)(v22 + 24) = v26 + 1;
          *v26 = v24;
          if ( !v23 )
            goto LABEL_27;
        }
        sub_16E7DE0(v22, v24);
      }
      while ( v23 );
LABEL_27:
      v27 = *(_QWORD **)(a1 + 8);
      v28 = *v8;
      v8 += 8;
      v29 = (*(__int64 (__fastcall **)(_QWORD *))(*v27 + 64LL))(v27);
      *(_QWORD *)(v28 + 184) = v29 + v27[3] - v27[1] - v32;
      sub_16E7EE0(*(_QWORD *)(a1 + 8), (char *)*(v8 - 3), *((unsigned int *)v8 - 4));
    }
    sub_39207C0(
      a1,
      *(_QWORD *)(a1 + 64),
      0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 72) - *(_QWORD *)(a1 + 64)) >> 3),
      v32);
    return sub_3919EA0(a1, v31);
  }
  return result;
}
