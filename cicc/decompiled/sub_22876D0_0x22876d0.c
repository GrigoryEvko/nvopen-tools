// Function: sub_22876D0
// Address: 0x22876d0
//
void __fastcall sub_22876D0(__int64 **a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 (*a5)(void))
{
  bool v8; // zf
  unsigned __int64 v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdi
  _QWORD *v13; // rdx
  __int64 v14; // rdi
  _WORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  unsigned __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  size_t v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  char v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h]

  v8 = *(_BYTE *)(a4 + 24) == 0;
  v24 = 0;
  if ( !v8 )
  {
    v16 = *(_QWORD *)(a4 + 16);
    v21 = 6;
    v22 = 0;
    v23 = v16;
    if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
      sub_BD6050(&v21, *(_QWORD *)a4 & 0xFFFFFFFFFFFFFFF8LL);
    v24 = 1;
  }
  v25 = *(_QWORD *)(a4 + 32);
  v9 = a5();
  if ( v24 )
  {
    v24 = 0;
    if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
      sub_BD60C0(&v21);
  }
  if ( v9 )
  {
    sub_2285FB0((__int64)&v21, a2, a4, (__int64 (__fastcall *)(unsigned __int64 *))a5, *a1[1]);
    v10 = (__int64)*a1;
    v11 = (*a1)[4];
    if ( (unsigned __int64)((*a1)[3] - v11) <= 4 )
    {
      v10 = sub_CB6200(v10, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v11 = 1685016073;
      *(_BYTE *)(v11 + 4) = 101;
      *(_QWORD *)(v10 + 32) += 5LL;
    }
    sub_CB5A80(v10, a2);
    v12 = (__int64)*a1;
    v13 = (_QWORD *)(*a1)[4];
    if ( (unsigned __int64)((*a1)[3] - (_QWORD)v13) <= 7 )
    {
      v12 = sub_CB6200(v12, " -> Node", 8u);
    }
    else
    {
      *v13 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v12 + 32) += 8LL;
    }
    sub_CB5A80(v12, v9);
    if ( v22 )
    {
      v17 = (__int64)*a1;
      v18 = (_BYTE *)(*a1)[4];
      if ( (_BYTE *)(*a1)[3] == v18 )
      {
        v17 = sub_CB6200(v17, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v18 = 91;
        ++*(_QWORD *)(v17 + 32);
      }
      v19 = sub_CB6200(v17, (unsigned __int8 *)v21, v22);
      v20 = *(_BYTE **)(v19 + 32);
      if ( *(_BYTE **)(v19 + 24) == v20 )
      {
        sub_CB6200(v19, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v20 = 93;
        ++*(_QWORD *)(v19 + 32);
      }
    }
    v14 = (__int64)*a1;
    v15 = (_WORD *)(*a1)[4];
    if ( (unsigned __int64)((*a1)[3] - (_QWORD)v15) <= 1 )
    {
      sub_CB6200(v14, (unsigned __int8 *)";\n", 2u);
    }
    else
    {
      *v15 = 2619;
      *(_QWORD *)(v14 + 32) += 2LL;
    }
    if ( (__int64 *)v21 != &v23 )
      j_j___libc_free_0(v21);
  }
}
