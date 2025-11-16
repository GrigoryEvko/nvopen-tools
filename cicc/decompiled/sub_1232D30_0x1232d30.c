// Function: sub_1232D30
// Address: 0x1232d30
//
__int64 __fastcall sub_1232D30(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int64 *v4; // rsi
  unsigned __int64 v6; // r15
  unsigned int v7; // r12d
  __int64 v9; // rdi
  unsigned int *v10; // r15
  __int64 v11; // r14
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  const char *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-98h]
  char v18; // [rsp+17h] [rbp-89h] BYREF
  __int64 v19; // [rsp+18h] [rbp-88h] BYREF
  unsigned int *v20; // [rsp+20h] [rbp-80h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  _BYTE v22[16]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v23[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v24; // [rsp+60h] [rbp-40h]

  v4 = &v19;
  v20 = (unsigned int *)v22;
  v6 = *(_QWORD *)(a1 + 232);
  v21 = 0x400000000LL;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v19, a3) )
    goto LABEL_2;
  v4 = (__int64 *)&v20;
  if ( (unsigned __int8)sub_120E620(a1, (__int64)&v20, &v18) )
    goto LABEL_2;
  v9 = *(_QWORD *)(v19 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 15 > 1 )
  {
    HIBYTE(v24) = 1;
    v16 = "extractvalue operand must be aggregate type";
LABEL_18:
    v4 = (__int64 *)v6;
    v23[0] = v16;
    LOBYTE(v24) = 3;
    sub_11FD800(a1 + 176, v6, (__int64)v23, 1);
LABEL_2:
    v7 = 1;
    goto LABEL_3;
  }
  if ( !sub_B501B0(v9, v20, (unsigned int)v21) )
  {
    HIBYTE(v24) = 1;
    v16 = "invalid indices for extractvalue";
    goto LABEL_18;
  }
  v10 = v20;
  v24 = 257;
  v17 = (unsigned int)v21;
  v11 = v19;
  v4 = (__int64 *)unk_3F10A14;
  v12 = sub_BD2C40(104, unk_3F10A14);
  if ( v12 )
  {
    v13 = sub_B501B0(*(_QWORD *)(v11 + 8), v10, v17);
    sub_B44260((__int64)v12, v13, 64, 1u, 0, 0);
    if ( *(v12 - 4) )
    {
      v14 = *(v12 - 3);
      *(_QWORD *)*(v12 - 2) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(v12 - 2);
    }
    *(v12 - 4) = v11;
    v15 = *(_QWORD *)(v11 + 16);
    *(v12 - 3) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = v12 - 3;
    *(v12 - 2) = v11 + 16;
    v4 = (__int64 *)v10;
    *(_QWORD *)(v11 + 16) = v12 - 4;
    v12[9] = v12 + 11;
    v12[10] = 0x400000000LL;
    sub_B50030((__int64)v12, v10, v17, (__int64)v23);
  }
  *a2 = v12;
  v7 = 2 * (v18 != 0);
LABEL_3:
  if ( v20 != (unsigned int *)v22 )
    _libc_free(v20, v4);
  return v7;
}
