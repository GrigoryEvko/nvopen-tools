// Function: sub_3026430
// Address: 0x3026430
//
void __fastcall sub_3026430(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rsi
  void *v6; // rbx
  int v7; // eax
  _DWORD *v8; // rax
  char *v9; // r15
  int v10; // r13d
  _WORD *v11; // rdx
  _QWORD *v12; // rax
  _DWORD *v13; // rax
  _QWORD *i; // rbx
  bool v15; // [rsp+Fh] [rbp-81h] BYREF
  _QWORD *v16; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-78h]
  void *v18; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v19; // [rsp+28h] [rbp-68h]
  _QWORD v20[2]; // [rsp+40h] [rbp-50h] BYREF
  int v21; // [rsp+50h] [rbp-40h]
  __int16 v22; // [rsp+54h] [rbp-3Ch]
  char v23; // [rsp+56h] [rbp-3Ah]

  v5 = (__int64 *)(a2 + 24);
  v6 = sub_C33340();
  if ( *(void **)(a2 + 24) == v6 )
  {
    sub_C3C790(&v18, (_QWORD **)v5);
    v7 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL);
    if ( v7 != 2 )
      goto LABEL_3;
LABEL_19:
    v13 = sub_C33310();
    v9 = "0f";
    v10 = 8;
    sub_C41640((__int64 *)&v18, v13, 1, &v15);
    goto LABEL_5;
  }
  sub_C33EB0(&v18, v5);
  v7 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL);
  if ( v7 == 2 )
    goto LABEL_19;
LABEL_3:
  if ( v7 != 3 )
    BUG();
  v8 = sub_C33320();
  v9 = "0d";
  v10 = 16;
  sub_C41640((__int64 *)&v18, v8, 1, &v15);
LABEL_5:
  if ( v6 == v18 )
    sub_C3E660((__int64)&v16, (__int64)&v18);
  else
    sub_C3A850((__int64)&v16, (__int64 *)&v18);
  v11 = *(_WORD **)(a3 + 32);
  if ( *(_QWORD *)(a3 + 24) - (_QWORD)v11 > 1u )
  {
    *v11 = *(_WORD *)v9;
    *(_QWORD *)(a3 + 32) += 2LL;
  }
  else
  {
    a3 = sub_CB6200(a3, (unsigned __int8 *)v9, 2u);
  }
  v12 = v16;
  if ( v17 > 0x40 )
    v12 = (_QWORD *)*v16;
  v20[0] = v12;
  v20[1] = 0;
  v21 = v10;
  v22 = 257;
  v23 = 0;
  sub_CB6AF0(a3, (__int64)v20);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0((unsigned __int64)v16);
  if ( v6 == v18 )
  {
    if ( v19 )
    {
      for ( i = &v19[3 * *(v19 - 1)]; v19 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v18);
  }
}
