// Function: sub_1530C90
// Address: 0x1530c90
//
void __fastcall sub_1530C90(__int64 a1)
{
  __int64 *v1; // r12
  __int64 *v2; // r14
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 *v9; // r14
  __int64 *v10; // r8
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 *v14; // r8
  unsigned __int8 *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // [rsp+0h] [rbp-110h]
  __int64 *v19; // [rsp+0h] [rbp-110h]
  __int64 *v20; // [rsp+8h] [rbp-108h]
  __int64 *v21; // [rsp+8h] [rbp-108h]
  __int64 *v22; // [rsp+10h] [rbp-100h]
  __int64 v23; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v24; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v26; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+48h] [rbp-C8h] BYREF
  unsigned __int64 v28; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v30[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v31; // [rsp+70h] [rbp-A0h]
  unsigned __int64 v32[2]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD v33[2]; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int8 *v34; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v35; // [rsp+A8h] [rbp-68h]
  _QWORD v36[12]; // [rsp+B0h] [rbp-60h] BYREF

  v1 = *(__int64 **)(a1 + 192);
  if ( *(__int64 **)(a1 + 184) == v1 )
  {
    v5 = *(_QWORD *)(a1 + 192);
LABEL_12:
    v35 = 0;
    v34 = (unsigned __int8 *)v36;
    *(_BYTE *)(a1 + 177) = 1;
    sub_168A3A0(&v23, v1, (v5 - (__int64)v1) >> 3, &v34, a1 + 16, a1 + 72);
    v6 = (_QWORD *)(v23 & 0xFFFFFFFFFFFFFFFELL);
    if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v23 = 0;
      v7 = *v6;
      v24 = 0;
      v25 = 0;
      v26 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(v7 + 48))(v6, &unk_4FA032A) )
      {
        v8 = (__int64 *)v6[1];
        v9 = (__int64 *)v6[2];
        v27 = 1;
        if ( v8 == v9 )
        {
          v13 = 1;
          v10 = v30;
        }
        else
        {
          v10 = v30;
          do
          {
            v18 = v8;
            v20 = v10;
            v30[0] = *v8;
            *v8 = 0;
            sub_15249C0(&v29, v10);
            v11 = v27;
            v27 = 0;
            v32[0] = v11 | 1;
            sub_12BEC00(&v28, v32, (unsigned __int64 *)&v29);
            if ( (v27 & 1) != 0 || (v10 = v20, v12 = v18, (v27 & 0xFFFFFFFFFFFFFFFELL) != 0) )
              sub_16BCAE0(&v27);
            v27 |= v28 | 1;
            if ( (v32[0] & 1) != 0 || (v32[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(v32);
            if ( (v29 & 1) != 0 || (v29 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(&v29);
            if ( v30[0] )
            {
              v19 = v20;
              v21 = v12;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v30[0] + 8LL))(v30[0]);
              v10 = v19;
              v12 = v21;
            }
            v8 = v12 + 1;
          }
          while ( v9 != v8 );
          v13 = v27 | 1;
        }
        v30[0] = v13;
        v22 = v10;
        (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
        v14 = v22;
      }
      else
      {
        v32[0] = (unsigned __int64)v6;
        sub_15249C0(v30, v32);
        v14 = v30;
        if ( v32[0] )
        {
          (*(void (__fastcall **)(unsigned __int64, unsigned __int64 *, __int64, __int64, __int64 *))(*(_QWORD *)v32[0] + 8LL))(
            v32[0],
            v32,
            v16,
            v17,
            v30);
          v14 = v30;
        }
      }
      v30[0] = ((v30[0] & 0xFFFFFFFFFFFFFFFELL) != 0) | v30[0] & 0xFFFFFFFFFFFFFFFELL;
      sub_14ECA90(v14);
      sub_14ECA90(&v26);
      sub_14ECA90(&v25);
      sub_14ECA90(&v24);
      sub_14ECA90(&v23);
      v15 = v34;
      if ( v34 == (unsigned __int8 *)v36 )
        return;
    }
    else
    {
      sub_152B430(a1, 0x19u, 1u, v34, (unsigned int)v35);
      v15 = v34;
      if ( v34 == (unsigned __int8 *)v36 )
        return;
    }
    _libc_free((unsigned __int64)v15);
    return;
  }
  v2 = *(__int64 **)(a1 + 184);
  while ( 1 )
  {
    v3 = *v2;
    if ( *(_QWORD *)(*v2 + 96) )
      break;
LABEL_10:
    if ( v1 == ++v2 )
    {
      v1 = *(__int64 **)(a1 + 184);
      v5 = *(_QWORD *)(a1 + 192);
      goto LABEL_12;
    }
  }
  LOBYTE(v33[0]) = 0;
  v31 = 260;
  v30[0] = v3 + 240;
  v32[0] = (unsigned __int64)v33;
  v32[1] = 0;
  sub_16E1010(&v34);
  v4 = sub_16D3AC0(&v34, v32);
  if ( v4 && *(_QWORD *)(v4 + 104) )
  {
    if ( v34 != (unsigned __int8 *)v36 )
      j_j___libc_free_0(v34, v36[0] + 1LL);
    if ( (_QWORD *)v32[0] != v33 )
      j_j___libc_free_0(v32[0], v33[0] + 1LL);
    goto LABEL_10;
  }
  if ( v34 != (unsigned __int8 *)v36 )
    j_j___libc_free_0(v34, v36[0] + 1LL);
  if ( (_QWORD *)v32[0] != v33 )
    j_j___libc_free_0(v32[0], v33[0] + 1LL);
}
