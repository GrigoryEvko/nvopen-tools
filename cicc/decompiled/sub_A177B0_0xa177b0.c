// Function: sub_A177B0
// Address: 0xa177b0
//
__int64 __fastcall sub_A177B0(__int64 a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v3; // r14
  char v6; // bl
  _QWORD *v7; // r15
  unsigned __int64 v8; // rax
  _QWORD *v9; // r15
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 result; // rax
  _WORD *v15; // r14
  char v16; // di
  int v17; // edx
  int v18; // esi
  __int16 v19; // bx
  unsigned __int64 v20; // r15
  __int64 v21; // rdi
  char v22; // al
  __int64 v23; // rdi
  __int64 v24; // rdi
  char v25; // al
  __int64 v26; // rdi
  char v27; // di
  int v28; // edx
  int v29; // esi
  __int16 v30; // bx
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+0h] [rbp-60h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  _WORD v36[25]; // [rsp+2Dh] [rbp-33h] BYREF

  v3 = a2 >> 3;
  v6 = a2 & 7;
  v7 = *(_QWORD **)(a1 + 32);
  if ( !v7 || !(unsigned __int8)sub_CB7440(*(_QWORD *)(a1 + 32)) )
    goto LABEL_18;
  if ( !(unsigned __int8)sub_CB7440(v7) )
    goto LABEL_35;
  v8 = v7[4] - v7[2] + (*(__int64 (__fastcall **)(_QWORD *))(*v7 + 80LL))(v7);
  v34 = v8;
  if ( v3 >= v8 )
  {
    v3 -= v8;
LABEL_18:
    result = *(_QWORD *)(a1 + 24);
    v15 = (_WORD *)(*(_QWORD *)result + v3);
    if ( (a2 & 7) != 0 )
    {
      v16 = 8 - v6;
      v17 = 1 << v6;
      v18 = (1 << v6) - 1;
      LOBYTE(v19) = *v15 & v18 | ((a3 & ((1 << (8 - v6)) - 1)) << v6);
      result = (unsigned __int8)HIBYTE(*v15);
      HIBYTE(v19) = ((int)(unsigned __int8)a3 >> v16) & v18 | result & -(char)v17;
      *v15 = v19;
    }
    else
    {
      *(_BYTE *)v15 = a3;
    }
    return result;
  }
  v9 = *(_QWORD **)(a1 + 32);
  if ( !v9 || !(unsigned __int8)sub_CB7440(*(_QWORD *)(a1 + 32)) )
LABEL_35:
    BUG();
  v33 = (*(__int64 (__fastcall **)(_QWORD *))(*v9 + 80LL))(v9) + v9[4] - v9[2];
  if ( (a2 & 7) != 0 )
  {
    v20 = v34;
    v35 = 1;
    v11 = v20 - v3;
    if ( v11 != 1 )
    {
      v35 = 0;
      v11 = 2;
    }
    v21 = *(_QWORD *)(a1 + 32);
    if ( !v21 || (v31 = *(_QWORD *)(a1 + 32), v22 = sub_CB7440(v21), v23 = v31, !v22) )
      v23 = 0;
    sub_CB70C0(v23, a2 >> 3);
    v24 = *(_QWORD *)(a1 + 32);
    if ( !v24 || (v32 = *(_QWORD *)(a1 + 32), v25 = sub_CB7440(v24), v26 = v32, !v25) )
      v26 = 0;
    sub_CB73E0(v26, v36, v11);
    if ( v35 )
      *((_BYTE *)v36 + v11) = ***(_BYTE ***)(a1 + 24);
    v27 = 8 - v6;
    v28 = 1 << v6;
    v29 = (1 << v6) - 1;
    LOBYTE(v30) = LOBYTE(v36[0]) & v29 | ((a3 & ((1 << (8 - v6)) - 1)) << v6);
    HIBYTE(v30) = ((int)(unsigned __int8)a3 >> v27) & v29 | HIBYTE(v36[0]) & -(char)v28;
    v36[0] = v30;
    v10 = *(_QWORD *)(a1 + 32);
    if ( v10 )
    {
LABEL_9:
      if ( (unsigned __int8)sub_CB7440(v10) )
        goto LABEL_10;
    }
  }
  else
  {
    LOBYTE(v36[0]) = a3;
    v10 = *(_QWORD *)(a1 + 32);
    v11 = 1;
    v35 = 0;
    if ( v10 )
      goto LABEL_9;
  }
  v10 = 0;
LABEL_10:
  sub_CB70C0(v10, v3);
  v12 = *(_QWORD *)(a1 + 32);
  if ( !v12 || !(unsigned __int8)sub_CB7440(*(_QWORD *)(a1 + 32)) )
    v12 = 0;
  sub_CB6200(v12, v36, v11);
  if ( v35 )
    ***(_BYTE ***)(a1 + 24) = *((_BYTE *)v36 + v11);
  v13 = *(_QWORD *)(a1 + 32);
  if ( !v13 || !(unsigned __int8)sub_CB7440(v13) )
    v13 = 0;
  return sub_CB70C0(v13, v33);
}
