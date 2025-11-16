// Function: sub_2BE5540
// Address: 0x2be5540
//
__int64 __fastcall sub_2BE5540(char *a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  char v5; // al
  __int64 v6; // r15
  unsigned int v7; // eax
  unsigned int v8; // r13d
  char v10; // dl
  size_t v11; // r14
  __int64 v12; // rax
  __int64 v13; // r9
  volatile signed __int32 *v14; // rax
  __int64 v15; // rax
  char **v16; // r15
  volatile signed __int32 **v17; // rsi
  unsigned int v18; // r13d
  __int64 v19; // r14
  char v20; // al
  unsigned int v21; // eax
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r14
  volatile signed __int32 *v27; // rax
  volatile signed __int32 *v28; // rdi
  char **v29; // [rsp+10h] [rbp-B0h]
  char v30; // [rsp+1Fh] [rbp-A1h]
  char v31; // [rsp+20h] [rbp-A0h]
  __int64 v32; // [rsp+20h] [rbp-A0h]
  __int64 v33; // [rsp+20h] [rbp-A0h]
  _BYTE *src; // [rsp+28h] [rbp-98h]
  char srca; // [rsp+28h] [rbp-98h]
  unsigned __int64 v36[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v37; // [rsp+40h] [rbp-80h] BYREF
  __int64 v38[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v39[2]; // [rsp+60h] [rbp-60h] BYREF
  volatile signed __int32 *v40; // [rsp+70h] [rbp-50h] BYREF
  size_t v41; // [rsp+78h] [rbp-48h]
  _QWORD v42[8]; // [rsp+80h] [rbp-40h] BYREF

  v3 = a1[8];
  v4 = sub_222F790(*(_QWORD **)(*(_QWORD *)a1 + 104LL), a2);
  v5 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 32LL))(v4, v3);
  v6 = *(_QWORD *)a1;
  LOBYTE(v40) = v5;
  LOBYTE(v7) = sub_2BE37E0(*(char **)v6, *(_BYTE **)(v6 + 8), (char *)&v40);
  v8 = v7;
  if ( (_BYTE)v7 )
    return v8;
  v10 = a1[8];
  v38[0] = (__int64)v39;
  sub_2240A50(v38, 1u, v10);
  v11 = v38[1];
  src = (_BYTE *)v38[0];
  v12 = sub_221F880(*(_QWORD **)(v6 + 104), 1);
  v36[0] = v11;
  v13 = v12;
  v40 = (volatile signed __int32 *)v42;
  if ( v11 > 0xF )
  {
    v32 = v12;
    v27 = (volatile signed __int32 *)sub_22409D0((__int64)&v40, v36, 0);
    v13 = v32;
    v40 = v27;
    v28 = v27;
    v42[0] = v36[0];
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(v42[0]) = *src;
      v14 = (volatile signed __int32 *)v42;
      goto LABEL_6;
    }
    if ( !v11 )
    {
      v14 = (volatile signed __int32 *)v42;
      goto LABEL_6;
    }
    v28 = (volatile signed __int32 *)v42;
  }
  v33 = v13;
  memcpy((void *)v28, src, v11);
  v11 = v36[0];
  v14 = v40;
  v13 = v33;
LABEL_6:
  v41 = v11;
  *((_BYTE *)v14 + v11) = 0;
  (*(void (__fastcall **)(unsigned __int64 *, __int64, volatile signed __int32 *, char *))(*(_QWORD *)v13 + 24LL))(
    v36,
    v13,
    v40,
    (char *)v40 + v41);
  if ( v40 != (volatile signed __int32 *)v42 )
    j_j___libc_free_0((unsigned __int64)v40);
  if ( (_QWORD *)v38[0] != v39 )
    j_j___libc_free_0(v38[0]);
  v15 = *(_QWORD *)a1;
  v29 = *(char ***)(*(_QWORD *)a1 + 56LL);
  if ( v29 != *(char ***)(*(_QWORD *)a1 + 48LL) )
  {
    v16 = *(char ***)(*(_QWORD *)a1 + 48LL);
    do
    {
      v17 = *(volatile signed __int32 ***)(v15 + 104);
      v18 = *(char *)v36[0];
      srca = *v16[4];
      v30 = **v16;
      sub_2208E20(&v40, v17);
      v19 = sub_222F790(&v40, (__int64)v17);
      sub_2209150(&v40);
      v31 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v19 + 32LL))(v19, v18);
      v20 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v19 + 16LL))(v19, v18);
      if ( v30 <= v31 && srca >= v31 )
        goto LABEL_14;
      if ( v30 <= v20 && srca >= v20 )
        goto LABEL_14;
      v15 = *(_QWORD *)a1;
      v16 += 8;
    }
    while ( v29 != v16 );
  }
  LOBYTE(v21) = sub_2BDBFE0(*(_QWORD **)(v15 + 112), (unsigned int)a1[8], *(_WORD *)(v15 + 96), *(_BYTE *)(v15 + 98));
  v8 = v21;
  if ( (_BYTE)v21 )
    goto LABEL_14;
  v22 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  sub_2BE40B0((__int64)&v40, *(_QWORD **)(*(_QWORD *)a1 + 112LL), a1 + 8, (__int64)(a1 + 9));
  v23 = sub_2BDD0F0(*(_QWORD *)(*(_QWORD *)a1 + 24LL), *(_QWORD *)(*(_QWORD *)a1 + 32LL), (__int64)&v40);
  if ( v40 != (volatile signed __int32 *)v42 )
    j_j___libc_free_0((unsigned __int64)v40);
  if ( v22 != v23 )
  {
LABEL_14:
    v8 = 1;
    goto LABEL_15;
  }
  v24 = *(_QWORD *)a1;
  v25 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
  v26 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( v26 != v25 )
  {
    while ( sub_2BDBFE0(*(_QWORD **)(v24 + 112), (unsigned int)a1[8], *(_WORD *)v25, *(_BYTE *)(v25 + 2)) )
    {
      v25 += 4;
      if ( v26 == v25 )
        goto LABEL_15;
      v24 = *(_QWORD *)a1;
    }
    goto LABEL_14;
  }
LABEL_15:
  if ( (__int64 *)v36[0] != &v37 )
    j_j___libc_free_0(v36[0]);
  return v8;
}
