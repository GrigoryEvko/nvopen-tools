// Function: sub_11E50A0
// Address: 0x11e50a0
//
__int64 __fastcall sub_11E50A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 *v6; // r12
  __int64 v7; // rdi
  _QWORD *v8; // r14
  int v9; // edx
  __int64 v10; // r13
  int v11; // edx
  unsigned int v12; // r9d
  __int16 v13; // ax
  char v14; // r12
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rax
  char v18; // r8
  __int64 v19; // r13
  _QWORD *v20; // rax
  __int64 v21; // r9
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // rdx
  _BYTE *v30; // rax
  _QWORD *i; // rbx
  __int64 v32; // rax
  __int64 v34; // [rsp+8h] [rbp-D8h]
  __int64 *v36; // [rsp+18h] [rbp-C8h]
  unsigned int v37; // [rsp+18h] [rbp-C8h]
  char v38; // [rsp+18h] [rbp-C8h]
  char v39; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+18h] [rbp-C8h]
  unsigned int v41; // [rsp+18h] [rbp-C8h]
  char v42; // [rsp+2Fh] [rbp-B1h] BYREF
  __int64 v43; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-A8h]
  char v45; // [rsp+3Ch] [rbp-A4h]
  _QWORD *v46; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v47; // [rsp+48h] [rbp-98h]
  __int64 v48[4]; // [rsp+60h] [rbp-80h] BYREF
  char v49[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v50; // [rsp+A0h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = (__int64 *)(v5 + 24);
  if ( *(_BYTE *)v5 != 18 )
  {
    v28 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v28 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v30 = sub_AD7630(v5, 0, v28);
    if ( !v30 || *v30 != 18 )
      return 0;
    v6 = (__int64 *)(v30 + 24);
    v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  v7 = *(_QWORD *)(a2 + 32 * (1 - v4));
  if ( *(_BYTE *)v7 != 18 )
  {
    v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v26 <= 1 && *(_BYTE *)v7 <= 0x15u )
    {
      v27 = sub_AD7630(v7, 0, v26);
      if ( v27 )
      {
        if ( *v27 == 18 )
        {
          v36 = (__int64 *)(v27 + 24);
          goto LABEL_4;
        }
      }
    }
    return 0;
  }
  v36 = (__int64 *)(v7 + 24);
LABEL_4:
  v8 = sub_C33340();
  if ( (_QWORD *)*v6 == v8 )
    sub_C3C790(&v46, (_QWORD **)v6);
  else
    sub_C33EB0(&v46, v6);
  if ( v46 == v8 )
    v9 = sub_C3EF50(&v46, (__int64)v36, 1u);
  else
    v9 = sub_C3B6C0((__int64)&v46, (__int64)v36, 1);
  v10 = 0;
  if ( (v9 & 0xFFFFFFEF) == 0 )
  {
    if ( v8 == (_QWORD *)*v6 )
      sub_C3C790(v48, (_QWORD **)v6);
    else
      sub_C33EB0(v48, v6);
    if ( v8 == (_QWORD *)v48[0] )
      v11 = sub_C3E9B0(v48, (__int64)v36);
    else
      v11 = sub_C3C0A0(v48, v36);
    v10 = 0;
    if ( !v11 )
    {
      v12 = *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL);
      v44 = v12;
      if ( v12 > 0x40 )
      {
        v41 = v12;
        sub_C43690((__int64)&v43, 0, 0);
        v12 = v41;
      }
      else
      {
        v43 = 0;
      }
      v10 = 0;
      v37 = v12;
      v45 = 0;
      if ( (sub_C41980((void **)&v46, (__int64)&v43, 1, &v42) & 0xFFFFFFEF) == 0 )
      {
        v13 = sub_A74840((_QWORD *)(a2 + 72), 2);
        v14 = HIBYTE(v13);
        v34 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( v45 )
        {
          v10 = v43;
          if ( v44 > 0x40 )
            v10 = *(_QWORD *)v43;
        }
        else if ( v44 > 0x40 )
        {
          v10 = *(_QWORD *)v43;
        }
        else if ( v44 )
        {
          v10 = v43 << (64 - (unsigned __int8)v44) >> (64 - (unsigned __int8)v44);
        }
        v15 = v37;
        v38 = v13;
        v16 = sub_BCD140((_QWORD *)a3[9], v15);
        v17 = sub_ACD640(v16, v10, 0);
        v18 = v38;
        v19 = v17;
        if ( !v14 )
        {
          v32 = sub_AA4E30(a3[6]);
          v18 = sub_AE5020(v32, *(_QWORD *)(v19 + 8));
        }
        v39 = v18;
        v50 = 257;
        v20 = sub_BD2C40(80, unk_3F10A10);
        v22 = (__int64)v20;
        if ( v20 )
          sub_B4D3C0((__int64)v20, v19, v34, 0, v39, v21, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
          a3[11],
          v22,
          v49,
          a3[7],
          a3[8]);
        v23 = *a3;
        v40 = *a3 + 16LL * *((unsigned int *)a3 + 2);
        if ( *a3 != v40 )
        {
          do
          {
            v24 = *(_QWORD *)(v23 + 8);
            v25 = *(_DWORD *)v23;
            v23 += 16;
            sub_B99FD0(v22, v25, v24);
          }
          while ( v40 != v23 );
        }
        v10 = sub_AD8F10(*(_QWORD *)(a2 + 8), v48);
      }
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
    }
    sub_91D830(v48);
  }
  if ( v8 == v46 )
  {
    if ( v47 )
    {
      for ( i = &v47[3 * *(v47 - 1)]; v47 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v46);
  }
  return v10;
}
