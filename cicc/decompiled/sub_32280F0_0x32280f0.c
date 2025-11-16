// Function: sub_32280F0
// Address: 0x32280f0
//
void __fastcall sub_32280F0(char a1, int a2, _QWORD *a3, __int64 *a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // r8
  __int64 *v7; // rax
  __int64 *v8; // rbx
  char *v9; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // r13
  _BYTE *v18; // rdi
  __int64 v19; // r15
  __int64 v20; // rdx
  char v21; // al
  char *v22; // r13
  __int64 *v25; // [rsp+10h] [rbp-150h]
  __int64 v26; // [rsp+28h] [rbp-138h]
  char *v27; // [rsp+28h] [rbp-138h]
  unsigned __int64 v28[2]; // [rsp+38h] [rbp-128h] BYREF
  _DWORD v29[4]; // [rsp+48h] [rbp-118h] BYREF
  char v30; // [rsp+58h] [rbp-108h]
  int v31; // [rsp+5Ch] [rbp-104h]
  char v32; // [rsp+78h] [rbp-E8h]
  __int64 v33; // [rsp+80h] [rbp-E0h]
  _BYTE *v34; // [rsp+88h] [rbp-D8h] BYREF
  __int64 v35; // [rsp+90h] [rbp-D0h]
  _BYTE v36[48]; // [rsp+98h] [rbp-C8h] BYREF
  char v37; // [rsp+C8h] [rbp-98h]
  int v38; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v39; // [rsp+D8h] [rbp-88h]
  _QWORD v40[2]; // [rsp+E0h] [rbp-80h] BYREF
  _BYTE v41[112]; // [rsp+F0h] [rbp-70h] BYREF

  v6 = (_BYTE *)(16 * a5);
  v7 = (__int64 *)&v6[(_QWORD)a4];
  v8 = a4;
  v9 = (char *)&v38;
  v25 = v7;
  if ( v8 != v7 )
  {
    while ( 1 )
    {
      v26 = *v8;
      if ( !a3 )
        break;
      v19 = v8[1];
      v20 = (__int64)a3;
      if ( !(unsigned int)((__int64)(*(_QWORD *)(v19 + 24) - *(_QWORD *)(v19 + 16)) >> 3) )
        goto LABEL_17;
      if ( !sub_AF46F0((__int64)a3) )
      {
        v20 = sub_32237B0(a3, v19);
        goto LABEL_17;
      }
LABEL_14:
      v8 += 2;
      if ( v25 == v8 )
        return;
    }
    v20 = 0;
LABEL_17:
    v33 = v20;
    v30 = a1;
    v31 = a2;
    v28[1] = 0x200000001LL;
    v35 = 0x200000000LL;
    v28[0] = (unsigned __int64)v29;
    v29[0] = 0;
    v32 = 0;
    v34 = v36;
    sub_3218BB0((__int64)&v34, (__int64)v28, v20, (__int64)v9, (__int64)v6, (__int64)&v34);
    v6 = v41;
    v21 = v32;
    v40[0] = v41;
    v39 = v33;
    v37 = v32;
    v38 = v26;
    v40[1] = 0x200000000LL;
    if ( (_DWORD)v35 )
    {
      sub_3218BB0((__int64)v40, (__int64)&v34, (unsigned int)v35, 0x200000000LL, (__int64)v41, (__int64)&v34);
      v21 = v37;
      v6 = v41;
    }
    v41[48] = v21;
    if ( v34 != v36 )
    {
      _libc_free((unsigned __int64)v34);
      v6 = v41;
    }
    v12 = *(unsigned int *)(a6 + 8);
    v13 = *(_QWORD *)a6;
    v9 = (char *)&v38;
    v14 = v12 + 1;
    v15 = *(_DWORD *)(a6 + 8);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
    {
      if ( v13 > (unsigned __int64)&v38 || (unsigned __int64)&v38 >= v13 + 88 * v12 )
      {
        sub_3227FC0(a6, v12 + 1, v12, (__int64)&v38, (__int64)v41, v14);
        v12 = *(unsigned int *)(a6 + 8);
        v13 = *(_QWORD *)a6;
        v9 = (char *)&v38;
        v6 = v41;
        v15 = *(_DWORD *)(a6 + 8);
      }
      else
      {
        v22 = (char *)&v38 - v13;
        sub_3227FC0(a6, v12 + 1, v12, (__int64)&v38 - v13, (__int64)v41, v14);
        v13 = *(_QWORD *)a6;
        v12 = *(unsigned int *)(a6 + 8);
        v6 = v41;
        v9 = &v22[*(_QWORD *)a6];
        v15 = *(_DWORD *)(a6 + 8);
      }
    }
    v16 = 11 * v12;
    v17 = v13 + 8 * v16;
    if ( v17 )
    {
      *(_DWORD *)v17 = *(_DWORD *)v9;
      *(_QWORD *)(v17 + 8) = *((_QWORD *)v9 + 1);
      *(_QWORD *)(v17 + 16) = v17 + 32;
      *(_QWORD *)(v17 + 24) = 0x200000000LL;
      if ( *((_DWORD *)v9 + 6) )
      {
        v27 = v9;
        sub_3218BB0(v17 + 16, (__int64)(v9 + 16), v16, (__int64)v9, (__int64)v41, v14);
        v6 = v41;
        v9 = v27;
      }
      *(_BYTE *)(v17 + 80) = v9[80];
      v15 = *(_DWORD *)(a6 + 8);
    }
    v18 = (_BYTE *)v40[0];
    *(_DWORD *)(a6 + 8) = v15 + 1;
    if ( v18 != v41 )
      _libc_free((unsigned __int64)v18);
    if ( (_DWORD *)v28[0] != v29 )
      _libc_free(v28[0]);
    goto LABEL_14;
  }
}
