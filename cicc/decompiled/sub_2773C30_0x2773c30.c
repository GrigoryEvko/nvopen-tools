// Function: sub_2773C30
// Address: 0x2773c30
//
__int64 __fastcall sub_2773C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 *v11; // r12
  void **v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  void **v16; // rax
  __int64 v17; // [rsp+8h] [rbp-E8h]
  __int64 v18[2]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 *v19; // [rsp+20h] [rbp-D0h]
  _BYTE v20[8]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+38h] [rbp-B8h]
  __int64 v22; // [rsp+40h] [rbp-B0h]
  __int64 v23; // [rsp+48h] [rbp-A8h]
  __int64 v24; // [rsp+50h] [rbp-A0h]
  __int64 *v25; // [rsp+58h] [rbp-98h]
  __int64 v26; // [rsp+60h] [rbp-90h] BYREF
  void **v27; // [rsp+68h] [rbp-88h]
  unsigned int v28; // [rsp+70h] [rbp-80h]
  unsigned int v29; // [rsp+74h] [rbp-7Ch]
  int v30; // [rsp+78h] [rbp-78h]
  char v31; // [rsp+7Ch] [rbp-74h]
  _QWORD v32[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v33; // [rsp+90h] [rbp-60h] BYREF
  _BYTE *v34; // [rsp+98h] [rbp-58h]
  __int64 v35; // [rsp+A0h] [rbp-50h]
  int v36; // [rsp+A8h] [rbp-48h]
  char v37; // [rsp+ACh] [rbp-44h]
  _BYTE v38[64]; // [rsp+B0h] [rbp-40h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v17 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v8 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  sub_1049690(v18, a3);
  v22 = v7;
  v21 = v6 + 8;
  v23 = v17 + 8;
  v24 = v8;
  v25 = v18;
  if ( !(unsigned __int8)sub_2770AC0((__int64)v20, a3) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_3;
  }
  v27 = (void **)v32;
  v28 = 2;
  v30 = 0;
  v31 = 1;
  v33 = 0;
  v34 = v38;
  v35 = 2;
  v36 = 0;
  v37 = 1;
  v29 = 1;
  v32[0] = &unk_4F81450;
  v26 = 1;
  if ( !v20[0] && !(unsigned __int8)sub_B19060((__int64)&v26, (__int64)&qword_4F82400, v9, v10) )
  {
    if ( !v31 )
    {
LABEL_20:
      sub_C8CC70((__int64)&v26, (__int64)&unk_4F875F0, (__int64)v13, v14, (__int64)&v26, v15);
      goto LABEL_7;
    }
    v16 = v27;
    v14 = v29;
    v13 = &v27[v29];
    if ( v27 == v13 )
    {
LABEL_19:
      if ( v29 >= v28 )
        goto LABEL_20;
      ++v29;
      *v13 = &unk_4F875F0;
      ++v26;
    }
    else
    {
      while ( *v16 != &unk_4F875F0 )
      {
        if ( v13 == ++v16 )
          goto LABEL_19;
      }
    }
  }
LABEL_7:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v32, (__int64)&v26);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v38, (__int64)&v33);
  if ( v37 )
  {
    if ( v31 )
      goto LABEL_3;
    goto LABEL_9;
  }
  _libc_free((unsigned __int64)v34);
  if ( !v31 )
LABEL_9:
    _libc_free((unsigned __int64)v27);
LABEL_3:
  v11 = v19;
  if ( v19 )
  {
    sub_FDC110(v19);
    j_j___libc_free_0((unsigned __int64)v11);
  }
  return a1;
}
