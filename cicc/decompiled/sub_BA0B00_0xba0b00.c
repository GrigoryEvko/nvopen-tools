// Function: sub_BA0B00
// Address: 0xba0b00
//
__int64 __fastcall sub_BA0B00(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r8
  int v6; // r13d
  int v7; // r13d
  int v8; // eax
  unsigned int v9; // r15d
  _QWORD *v10; // r11
  int v11; // r9d
  bool v12; // r10
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // rdx
  __int64 v16; // r12
  unsigned int v17; // eax
  unsigned int v19; // esi
  int v20; // eax
  _QWORD *v21; // rdx
  int v22; // eax
  char v23; // al
  __int64 v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-98h]
  _QWORD *v26; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+10h] [rbp-90h]
  bool v29; // [rsp+1Fh] [rbp-81h]
  bool v30; // [rsp+1Fh] [rbp-81h]
  __int64 *v31; // [rsp+20h] [rbp-80h]
  __int64 *v32; // [rsp+20h] [rbp-80h]
  int v33; // [rsp+28h] [rbp-78h]
  int v34; // [rsp+28h] [rbp-78h]
  unsigned int v35; // [rsp+2Ch] [rbp-74h]
  __int64 v36; // [rsp+30h] [rbp-70h]
  __int64 v37; // [rsp+30h] [rbp-70h]
  __int64 v38; // [rsp+30h] [rbp-70h]
  __int64 v39[2]; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v40; // [rsp+48h] [rbp-58h] BYREF
  _QWORD *v41; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-48h]
  __int64 v43; // [rsp+60h] [rbp-40h] BYREF
  bool v44; // [rsp+68h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 24);
  v39[0] = a1;
  v42 = v3;
  if ( v3 > 0x40 )
    sub_C43780(&v41, a1 + 16);
  else
    v41 = *(_QWORD **)(a1 + 16);
  v4 = sub_AF5140(a1, 0);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_DWORD *)(a2 + 24);
  v43 = v4;
  v36 = v5;
  v44 = *(_DWORD *)(a1 + 4) != 0;
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = sub_AFB7E0((__int64)&v41, &v43);
    v9 = v42;
    v10 = v41;
    v11 = 1;
    v12 = v44;
    v13 = v43;
    v35 = v7 & v8;
    v14 = v36;
    while ( 1 )
    {
      v15 = (__int64 *)(v14 + 8LL * v35);
      v16 = *v15;
      if ( *v15 == -4096 )
        goto LABEL_11;
      if ( v16 != -8192 && v9 == *(_DWORD *)(v16 + 24) )
      {
        if ( v9 <= 0x40 )
        {
          if ( v10 != *(_QWORD **)(v16 + 16) )
            goto LABEL_8;
        }
        else
        {
          v25 = v10;
          v27 = v13;
          v29 = v12;
          v33 = v11;
          v31 = (__int64 *)(v14 + 8LL * v35);
          v37 = v14;
          v23 = sub_C43C50(&v41, v16 + 16);
          v14 = v37;
          v15 = v31;
          v11 = v33;
          v12 = v29;
          v13 = v27;
          v10 = v25;
          if ( !v23 )
            goto LABEL_8;
        }
        if ( v12 == (*(_DWORD *)(v16 + 4) != 0) )
        {
          v26 = v10;
          v28 = v13;
          v30 = v12;
          v34 = v11;
          v32 = v15;
          v38 = v14;
          v24 = sub_AF5140(v16, 0);
          v13 = v28;
          v14 = v38;
          v11 = v34;
          v12 = v30;
          v10 = v26;
          if ( v28 == v24 )
          {
            if ( v32 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
              goto LABEL_11;
            if ( v9 <= 0x40 )
              goto LABEL_15;
            goto LABEL_13;
          }
        }
      }
LABEL_8:
      v17 = v11 + v35;
      ++v11;
      v35 = v7 & v17;
    }
  }
  v9 = v42;
LABEL_11:
  if ( v9 <= 0x40 )
    goto LABEL_37;
  v16 = 0;
LABEL_13:
  if ( v41 )
    j_j___libc_free_0_0(v41);
LABEL_15:
  if ( !v16 )
  {
LABEL_37:
    if ( (unsigned __int8)sub_AFCD70(a2, v39, &v40) )
      return v39[0];
    v19 = *(_DWORD *)(a2 + 24);
    v20 = *(_DWORD *)(a2 + 16);
    v21 = v40;
    ++*(_QWORD *)a2;
    v22 = v20 + 1;
    v41 = v21;
    if ( 4 * v22 >= 3 * v19 )
    {
      v19 *= 2;
    }
    else if ( v19 - *(_DWORD *)(a2 + 20) - v22 > v19 >> 3 )
    {
LABEL_21:
      *(_DWORD *)(a2 + 16) = v22;
      if ( *v21 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v21 = v39[0];
      return v39[0];
    }
    sub_B04210(a2, v19);
    sub_AFCD70(a2, v39, &v41);
    v21 = v41;
    v22 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_21;
  }
  return v16;
}
