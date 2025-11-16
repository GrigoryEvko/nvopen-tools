// Function: sub_30583A0
// Address: 0x30583a0
//
void __fastcall sub_30583A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rsi
  void *v4; // rbx
  unsigned int v5; // eax
  bool v6; // cc
  _WORD *v7; // rdx
  _DWORD *v8; // rax
  int v9; // r13d
  _WORD *v10; // rdx
  _DWORD *v11; // rax
  _WORD *v12; // rdx
  _WORD *v13; // rdx
  _DWORD *v14; // rax
  _QWORD *v15; // rax
  _QWORD *i; // rbx
  bool v17; // [rsp+Fh] [rbp-71h] BYREF
  _QWORD *v18; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-68h]
  void *v20; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v21; // [rsp+28h] [rbp-58h]
  _QWORD v22[2]; // [rsp+40h] [rbp-40h] BYREF
  int v23; // [rsp+50h] [rbp-30h]
  __int16 v24; // [rsp+54h] [rbp-2Ch]
  char v25; // [rsp+56h] [rbp-2Ah]

  v3 = (__int64 *)(a1 + 32);
  v4 = sub_C33340();
  if ( *(void **)(a1 + 32) != v4 )
  {
    sub_C33EB0(&v20, v3);
    v5 = *(_DWORD *)(a1 + 24);
    v6 = v5 <= 3;
    if ( v5 != 3 )
      goto LABEL_3;
LABEL_19:
    v13 = *(_WORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 <= 1u )
    {
      sub_CB6200(a2, (unsigned __int8 *)"0f", 2u);
    }
    else
    {
      *v13 = 26160;
      *(_QWORD *)(a2 + 32) += 2LL;
    }
    v14 = sub_C33310();
    v9 = 8;
    sub_C41640((__int64 *)&v20, v14, 1, &v17);
    goto LABEL_22;
  }
  sub_C3C790(&v20, (_QWORD **)v3);
  v5 = *(_DWORD *)(a1 + 24);
  v6 = v5 <= 3;
  if ( v5 == 3 )
    goto LABEL_19;
LABEL_3:
  if ( !v6 )
  {
    if ( v5 != 4 )
      goto LABEL_40;
    v10 = *(_WORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v10 <= 1u )
    {
      sub_CB6200(a2, (unsigned __int8 *)"0d", 2u);
    }
    else
    {
      *v10 = 25648;
      *(_QWORD *)(a2 + 32) += 2LL;
    }
    v11 = sub_C33320();
    v9 = 16;
    sub_C41640((__int64 *)&v20, v11, 1, &v17);
LABEL_22:
    if ( v4 == v20 )
      goto LABEL_9;
    goto LABEL_23;
  }
  if ( v5 == 1 )
  {
    v7 = *(_WORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 1u )
    {
      sub_CB6200(a2, (unsigned __int8 *)"0x", 2u);
    }
    else
    {
      *v7 = 30768;
      *(_QWORD *)(a2 + 32) += 2LL;
    }
    v8 = sub_C33300();
    goto LABEL_8;
  }
  if ( v5 != 2 )
LABEL_40:
    BUG();
  v12 = *(_WORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 1u )
  {
    sub_CB6200(a2, (unsigned __int8 *)"0x", 2u);
  }
  else
  {
    *v12 = 30768;
    *(_QWORD *)(a2 + 32) += 2LL;
  }
  v8 = sub_C332F0();
LABEL_8:
  sub_C41640((__int64 *)&v20, v8, 1, &v17);
  v9 = 4;
  if ( v4 == v20 )
  {
LABEL_9:
    sub_C3E660((__int64)&v18, (__int64)&v20);
    goto LABEL_24;
  }
LABEL_23:
  sub_C3A850((__int64)&v18, (__int64 *)&v20);
LABEL_24:
  v15 = v18;
  if ( v19 > 0x40 )
    v15 = (_QWORD *)*v18;
  v22[0] = v15;
  v22[1] = 0;
  v23 = v9;
  v24 = 257;
  v25 = 0;
  sub_CB6AF0(a2, (__int64)v22);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0((unsigned __int64)v18);
  if ( v4 == v20 )
  {
    if ( v21 )
    {
      for ( i = &v21[3 * *(v21 - 1)]; v21 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v20);
  }
}
