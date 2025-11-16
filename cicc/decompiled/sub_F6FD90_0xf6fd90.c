// Function: sub_F6FD90
// Address: 0xf6fd90
//
__int64 __fastcall sub_F6FD90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // r10
  _QWORD *v15; // rax
  _QWORD *v16; // r10
  _QWORD **v17; // rdx
  int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rax
  unsigned int *v21; // r15
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // [rsp+0h] [rbp-B0h]
  __int64 v25; // [rsp+8h] [rbp-A8h]
  __int64 v26; // [rsp+8h] [rbp-A8h]
  __int64 v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+18h] [rbp-98h]
  const char *v30; // [rsp+20h] [rbp-90h] BYREF
  char v31; // [rsp+40h] [rbp-70h]
  char v32; // [rsp+41h] [rbp-6Fh]
  unsigned __int64 v33; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+58h] [rbp-58h]
  __int64 v35; // [rsp+60h] [rbp-50h]
  __int16 v36; // [rsp+70h] [rbp-40h]

  v5 = a2;
  v6 = *(_QWORD *)(a3 + 24);
  v33 = 6;
  v34 = 0;
  v35 = v6;
  if ( v6 == -4096 || v6 == 0 || v6 == -8192 )
  {
    v7 = v6;
  }
  else
  {
    sub_BD6050(&v33, *(_QWORD *)(a3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    v6 = v35;
    if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
      sub_BD60C0(&v33);
    v7 = *(_QWORD *)(a3 + 24);
  }
  v8 = *(_QWORD *)(v7 + 8);
  v9 = *(_DWORD *)(v8 + 8) >> 8;
  LODWORD(v34) = v9;
  if ( v9 > 0x40 )
  {
    v25 = v8;
    sub_C43690((__int64)&v33, 0, 0);
    v8 = v25;
    v10 = 1LL << ((unsigned __int8)v9 - 1);
    if ( (unsigned int)v34 > 0x40 )
    {
      *(_QWORD *)(v33 + 8LL * ((v9 - 1) >> 6)) |= v10;
      goto LABEL_11;
    }
  }
  else
  {
    v33 = 0;
    v10 = 1LL << ((unsigned __int8)v9 - 1);
  }
  v33 |= v10;
LABEL_11:
  v11 = sub_AD8D80(v8, (__int64)&v33);
  if ( (unsigned int)v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1 )
    v5 = sub_B34880(a1, a2, 1);
  v12 = *(_QWORD *)(a1 + 80);
  v32 = 1;
  v31 = 3;
  v30 = "rdx.select.cmp";
  v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v12 + 56LL))(v12, 33, v5, v11);
  if ( !v13 )
  {
    v36 = 257;
    v15 = sub_BD2C40(72, unk_3F10FD0);
    v16 = v15;
    if ( v15 )
    {
      v17 = *(_QWORD ***)(v5 + 8);
      v26 = (__int64)v15;
      v18 = *((unsigned __int8 *)v17 + 8);
      if ( (unsigned int)(v18 - 17) > 1 )
      {
        v20 = sub_BCB2A0(*v17);
      }
      else
      {
        BYTE4(v29) = (_BYTE)v18 == 18;
        LODWORD(v29) = *((_DWORD *)v17 + 8);
        v19 = (__int64 *)sub_BCB2A0(*v17);
        v20 = sub_BCE1B0(v19, v29);
      }
      sub_B523C0(v26, v20, 53, 33, v5, v11, (__int64)&v33, 0, 0, 0);
      v16 = (_QWORD *)v26;
    }
    v27 = (__int64)v16;
    (*(void (__fastcall **)(_QWORD, _QWORD *, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v16,
      &v30,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v13 = v27;
    v21 = *(unsigned int **)a1;
    v24 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v24 )
    {
      do
      {
        v22 = *((_QWORD *)v21 + 1);
        v23 = *v21;
        v21 += 4;
        v28 = v13;
        sub_B99FD0(v13, v23, v22);
        v13 = v28;
      }
      while ( (unsigned int *)v24 != v21 );
    }
  }
  v33 = (unsigned __int64)"rdx.select";
  v36 = 259;
  return sub_B36550((unsigned int **)a1, v13, v5, v6, (__int64)&v33, 0);
}
