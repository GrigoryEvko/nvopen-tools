// Function: sub_318E150
// Address: 0x318e150
//
void __fastcall sub_318E150(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r15
  unsigned __int64 v3; // rcx
  char **v4; // r8
  __int64 v5; // r9
  char *v6; // rbx
  char *v7; // r12
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 *v12; // r13
  char *v13; // rsi
  char *v14; // r15
  __int64 *v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rdx
  _QWORD *v22; // r13
  _BYTE *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  char *v27; // r13
  __int64 v28; // [rsp+0h] [rbp-120h]
  _QWORD *v29; // [rsp+8h] [rbp-118h]
  char *v30; // [rsp+10h] [rbp-110h]
  _QWORD *v31; // [rsp+30h] [rbp-F0h]
  char **v32; // [rsp+30h] [rbp-F0h]
  char *v33; // [rsp+40h] [rbp-E0h] BYREF
  int v34; // [rsp+48h] [rbp-D8h]
  char v35; // [rsp+50h] [rbp-D0h] BYREF
  char *v36; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+68h] [rbp-B8h]
  char v38; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD v39[2]; // [rsp+A0h] [rbp-80h] BYREF
  _BYTE v40[48]; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v41; // [rsp+E0h] [rbp-40h]

  v2 = a1;
  a1[12] = 0;
  *a1 = &unk_4A346C0;
  a1[1] = a1 + 3;
  a1[2] = 0x100000000LL;
  a1[13] = *a2;
  *a2 = 0;
  v29 = (_QWORD *)a1[13];
  (*(void (__fastcall **)(char **))(*v29 + 80LL))(&v33);
  v6 = &v33[8 * v34];
  v30 = v33;
  if ( v6 != v33 )
  {
    v7 = &v38;
    v28 = (__int64)(a1 + 1);
    do
    {
      v8 = *((_QWORD *)v6 - 1);
      v36 = v7;
      v37 = 0x600000000LL;
      v9 = 0;
      v10 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
      if ( v10 > 6 )
      {
        sub_C8D5F0((__int64)&v36, v7, v10, 8u, (__int64)v4, v5);
        v9 = (unsigned int)v37;
        v10 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
      }
      v11 = 32 * v10;
      if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
      {
        v12 = *(__int64 **)(v8 - 8);
        v5 = (__int64)v12 + v11;
      }
      else
      {
        v5 = v8;
        v12 = (__int64 *)(v8 - v11);
      }
      if ( v12 != (__int64 *)v5 )
      {
        v31 = v2;
        v13 = v7;
        v14 = v6;
        v15 = (__int64 *)v5;
        do
        {
          v3 = HIDWORD(v37);
          v16 = *v12;
          if ( v9 + 1 > (unsigned __int64)HIDWORD(v37) )
          {
            sub_C8D5F0((__int64)&v36, v13, v9 + 1, 8u, (__int64)v4, v5);
            v9 = (unsigned int)v37;
          }
          v11 = (__int64)v36;
          v12 += 4;
          *(_QWORD *)&v36[8 * v9] = v16;
          v9 = (unsigned int)(v37 + 1);
          LODWORD(v37) = v37 + 1;
        }
        while ( v15 != v12 );
        v6 = v14;
        v2 = v31;
        v7 = v13;
      }
      v39[1] = 0x600000000LL;
      v39[0] = v40;
      if ( (_DWORD)v9 )
        sub_318D930((__int64)v39, (__int64)&v36, v11, v3, (__int64)v4, v5);
      v17 = *((unsigned int *)v2 + 4);
      v18 = *((unsigned int *)v2 + 5);
      v41 = v8;
      v3 = v2[1];
      v4 = (char **)v39;
      v19 = v17 + 1;
      v20 = v17;
      if ( v17 + 1 > v18 )
      {
        if ( v3 > (unsigned __int64)v39 || (unsigned __int64)v39 >= v3 + 72 * v17 )
        {
          sub_318E050(v28, v19, v17, v3, (__int64)v39, v5);
          v17 = *((unsigned int *)v2 + 4);
          v3 = v2[1];
          v4 = (char **)v39;
          v20 = *((_DWORD *)v2 + 4);
        }
        else
        {
          v27 = (char *)v39 - v3;
          sub_318E050(v28, v19, v17, v3, (__int64)v39, v5);
          v3 = v2[1];
          v17 = *((unsigned int *)v2 + 4);
          v4 = (char **)&v27[v3];
          v20 = *((_DWORD *)v2 + 4);
        }
      }
      v21 = 9 * v17;
      v22 = (_QWORD *)(v3 + 8 * v21);
      if ( v22 )
      {
        *v22 = v22 + 2;
        v22[1] = 0x600000000LL;
        if ( *((_DWORD *)v4 + 2) )
        {
          v32 = v4;
          sub_318D7D0(v3 + 8 * v21, v4, v21, v3, (__int64)v4, v5);
          v4 = v32;
        }
        v22[8] = v4[8];
        v20 = *((_DWORD *)v2 + 4);
      }
      v23 = (_BYTE *)v39[0];
      *((_DWORD *)v2 + 4) = v20 + 1;
      if ( v23 != v40 )
        _libc_free((unsigned __int64)v23);
      if ( v36 != v7 )
        _libc_free((unsigned __int64)v36);
      v6 -= 8;
    }
    while ( v30 != v6 );
    v30 = v33;
  }
  v24 = v29[2];
  v25 = *(_QWORD *)(v24 + 40);
  v26 = *(_QWORD *)(v24 + 32);
  if ( !v26 || v26 == v25 + 48 )
    v2[12] = v25 | 4;
  else
    v2[12] = (v26 - 24) & 0xFFFFFFFFFFFFFFFBLL;
  if ( v30 != &v35 )
    _libc_free((unsigned __int64)v30);
}
