// Function: sub_30CAD10
// Address: 0x30cad10
//
__int64 *__fastcall sub_30CAD10(__int64 *a1, __int64 a2)
{
  _DWORD *v3; // rcx
  unsigned __int64 v4; // rax
  _QWORD *v5; // r15
  __int64 v6; // rax
  void *v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rdi
  _BYTE *v10; // rax
  char *v11; // rbx
  _QWORD *v12; // r15
  size_t v13; // rax
  unsigned __int64 *v14; // rax
  _DWORD *v16; // rdx
  unsigned __int64 v17[2]; // [rsp+0h] [rbp-130h] BYREF
  _BYTE v18[16]; // [rsp+10h] [rbp-120h] BYREF
  _QWORD v19[3]; // [rsp+20h] [rbp-110h] BYREF
  __int64 v20; // [rsp+38h] [rbp-F8h]
  _DWORD *v21; // [rsp+40h] [rbp-F0h]
  __int64 v22; // [rsp+48h] [rbp-E8h]
  unsigned __int64 *v23; // [rsp+50h] [rbp-E0h]
  __int64 *v24; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int8 *v26; // [rsp+80h] [rbp-B0h]
  size_t v27; // [rsp+88h] [rbp-A8h]
  __int64 v28; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v29[2]; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD v30[2]; // [rsp+C0h] [rbp-70h] BYREF
  unsigned __int8 *v31; // [rsp+D0h] [rbp-60h]
  size_t v32; // [rsp+D8h] [rbp-58h]
  _QWORD v33[10]; // [rsp+E0h] [rbp-50h] BYREF

  v22 = 0x100000000LL;
  v17[0] = (unsigned __int64)v18;
  v17[1] = 0;
  v19[0] = &unk_49DD210;
  v18[0] = 0;
  v19[1] = 0;
  v19[2] = 0;
  v20 = 0;
  v21 = 0;
  v23 = v17;
  sub_CB5980((__int64)v19, 0, 0, 0);
  if ( *(_DWORD *)a2 == 0x80000000 )
  {
    v16 = v21;
    if ( (unsigned __int64)(v20 - (_QWORD)v21) <= 0xC )
    {
      sub_CB6200((__int64)v19, "(cost=always)", 0xDu);
    }
    else
    {
      v21[2] = 1937334647;
      *(_QWORD *)v16 = 0x6C613D74736F6328LL;
      *((_BYTE *)v16 + 12) = 41;
      v21 = (_DWORD *)((char *)v21 + 13);
    }
  }
  else
  {
    v3 = v21;
    v4 = v20 - (_QWORD)v21;
    if ( *(_DWORD *)a2 == 0x7FFFFFFF )
    {
      if ( v4 <= 0xB )
      {
        sub_CB6200((__int64)v19, "(cost=never)", 0xCu);
      }
      else
      {
        v21[2] = 695362934;
        *(_QWORD *)v3 = 0x656E3D74736F6328LL;
        v21 += 3;
      }
    }
    else
    {
      if ( v4 <= 5 )
      {
        v5 = (_QWORD *)sub_CB6200((__int64)v19, "(cost=", 6u);
      }
      else
      {
        *v21 = 1936679720;
        v5 = v19;
        *((_WORD *)v3 + 2) = 15732;
        v21 = (_DWORD *)((char *)v21 + 6);
      }
      sub_B16530((__int64 *)&v24, "Cost", 4, *(_DWORD *)a2);
      v6 = sub_CB6200((__int64)v5, v26, v27);
      v7 = *(void **)(v6 + 32);
      v8 = v6;
      if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0xBu )
      {
        v8 = sub_CB6200(v6, ", threshold=", 0xCu);
      }
      else
      {
        qmemcpy(v7, ", threshold=", 12);
        *(_QWORD *)(v6 + 32) += 12LL;
      }
      sub_B16530(v29, "Threshold", 9, *(_DWORD *)(a2 + 4));
      v9 = sub_CB6200(v8, v31, v32);
      v10 = *(_BYTE **)(v9 + 32);
      if ( *(_BYTE **)(v9 + 24) == v10 )
      {
        sub_CB6200(v9, (unsigned __int8 *)")", 1u);
      }
      else
      {
        *v10 = 41;
        ++*(_QWORD *)(v9 + 32);
      }
      if ( v31 != (unsigned __int8 *)v33 )
        j_j___libc_free_0((unsigned __int64)v31);
      if ( (_QWORD *)v29[0] != v30 )
        j_j___libc_free_0(v29[0]);
      if ( v26 != (unsigned __int8 *)&v28 )
        j_j___libc_free_0((unsigned __int64)v26);
      if ( v24 != &v25 )
        j_j___libc_free_0((unsigned __int64)v24);
    }
  }
  v11 = *(char **)(a2 + 16);
  if ( v11 )
  {
    if ( (unsigned __int64)(v20 - (_QWORD)v21) <= 1 )
    {
      v12 = (_QWORD *)sub_CB6200((__int64)v19, (unsigned __int8 *)": ", 2u);
    }
    else
    {
      v12 = v19;
      *(_WORD *)v21 = 8250;
      v21 = (_DWORD *)((char *)v21 + 2);
    }
    v13 = strlen(v11);
    sub_B16430((__int64)v29, "Reason", 6u, v11, v13);
    sub_CB6200((__int64)v12, v31, v32);
    if ( v31 != (unsigned __int8 *)v33 )
      j_j___libc_free_0((unsigned __int64)v31);
    if ( (_QWORD *)v29[0] != v30 )
      j_j___libc_free_0(v29[0]);
  }
  v14 = v23;
  *a1 = (__int64)(a1 + 2);
  sub_30CA4D0(a1, (_BYTE *)*v14, *v14 + v14[1]);
  v19[0] = &unk_49DD210;
  sub_CB5840((__int64)v19);
  if ( (_BYTE *)v17[0] != v18 )
    j_j___libc_free_0(v17[0]);
  return a1;
}
