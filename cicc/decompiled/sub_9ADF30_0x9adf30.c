// Function: sub_9ADF30
// Address: 0x9adf30
//
__int64 __fastcall sub_9ADF30(
        unsigned __int8 *a1,
        unsigned __int8 *a2,
        bool a3,
        char a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        int a8,
        __m128i *a9)
{
  bool v11; // r10
  unsigned int v12; // eax
  unsigned __int64 v13; // rsi
  unsigned int v14; // r11d
  unsigned int v15; // ecx
  unsigned __int64 v16; // rdx
  unsigned int v17; // edi
  unsigned int v18; // r13d
  __int64 v19; // r8
  bool v20; // zf
  unsigned int v21; // ebx
  __int64 v22; // r8
  bool v23; // r14
  __int64 v24; // r13
  bool v25; // r13
  unsigned int v26; // r11d
  bool v27; // cc
  unsigned int v28; // eax
  __int64 v29; // rdi
  __int64 result; // rax
  unsigned int v31; // esi
  unsigned int v32; // esi
  unsigned __int64 v33; // rdx
  bool v34; // bl
  unsigned int v35; // ebx
  bool v36; // bl
  unsigned int v37; // esi
  unsigned __int64 v38; // rdx
  unsigned int v39; // ecx
  unsigned int v40; // esi
  bool v41; // r10
  unsigned __int16 v42; // ax
  char v43; // dl
  __int16 v44; // ax
  unsigned int v45; // ebx
  bool v46; // bl
  bool v48; // [rsp+10h] [rbp-80h]
  bool v49; // [rsp+10h] [rbp-80h]
  bool v50; // [rsp+10h] [rbp-80h]
  bool v51; // [rsp+10h] [rbp-80h]
  char v53; // [rsp+18h] [rbp-78h]
  char v54; // [rsp+18h] [rbp-78h]
  bool v55; // [rsp+20h] [rbp-70h]
  bool v56; // [rsp+28h] [rbp-68h]
  bool v57; // [rsp+28h] [rbp-68h]
  __int64 v58; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-58h]
  __int64 v60; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v61; // [rsp+48h] [rbp-48h]
  __int64 v62; // [rsp+50h] [rbp-40h]
  unsigned int v63; // [rsp+58h] [rbp-38h]

  v56 = a3;
  sub_9AB8E0(a2, a5, (unsigned __int64 *)a6, a8 + 1, a9);
  sub_9AB8E0(a1, a5, (unsigned __int64 *)a7, a8 + 1, a9);
  v11 = a3;
  if ( !a3 )
  {
    v25 = 0;
    v26 = 0;
    if ( a2 != a1 )
      goto LABEL_15;
LABEL_34:
    v56 = 0;
    v26 = (unsigned __int8)sub_98EF80(a1, a9[2].m128i_i64[0], a9[2].m128i_i64[1], a9[1].m128i_i64[1], a8 + 1);
    v25 = a3;
    goto LABEL_15;
  }
  if ( a2 == a1 )
    goto LABEL_34;
  v12 = *((_DWORD *)a6 + 2);
  v13 = *a6;
  v14 = v12 - 1;
  if ( v12 > 0x40 )
    v13 = *(_QWORD *)(v13 + 8LL * (v14 >> 6));
  v15 = *(_DWORD *)(a7 + 8);
  v16 = *(_QWORD *)a7;
  v17 = v15 - 1;
  if ( v15 > 0x40 )
    v16 = *(_QWORD *)(v16 + 8LL * (v17 >> 6));
  v18 = *((_DWORD *)a6 + 6);
  v19 = a6[2];
  if ( v18 > 0x40 )
    v19 = *(_QWORD *)(v19 + 8LL * ((v18 - 1) >> 6));
  v20 = (v19 & (1LL << ((unsigned __int8)v18 - 1))) == 0;
  v21 = *(_DWORD *)(a7 + 24);
  v22 = *(_QWORD *)(a7 + 16);
  v23 = !v20;
  v24 = 1LL << ((unsigned __int8)v21 - 1);
  if ( v21 > 0x40 )
    v22 = *(_QWORD *)(v22 + 8LL * ((v21 - 1) >> 6));
  v55 = (v22 & v24) != 0;
  if ( (v22 & v24) != 0 && !v20 )
    goto LABEL_13;
  v57 = (v13 & (1LL << v14)) != 0;
  v34 = (v16 & (1LL << v17)) != 0;
  v25 = v34 && v57;
  if ( v34 && v57 )
    goto LABEL_13;
  if ( !a4 )
    goto LABEL_37;
  v59 = *((_DWORD *)a6 + 2);
  if ( v12 > 0x40 )
  {
    sub_C43690(&v58, 1, 0);
    v11 = a3;
  }
  else
  {
    v58 = 1;
  }
  v48 = v11;
  sub_987BA0((__int64)&v60, &v58);
  v41 = v48;
  if ( v59 > 0x40 && v58 )
  {
    j_j___libc_free_0_0(v58);
    v41 = v48;
  }
  v49 = v41;
  v42 = sub_C77470(a6, &v60);
  v11 = v49;
  if ( !HIBYTE(v42) || (v43 = v42) == 0 )
  {
    v44 = sub_C77470(a7, &v60);
    v11 = v49;
    v43 = HIBYTE(v44);
    if ( HIBYTE(v44) )
      v43 = v44;
  }
  if ( v63 > 0x40 && v62 )
  {
    v50 = v11;
    v53 = v43;
    j_j___libc_free_0_0(v62);
    v11 = v50;
    v43 = v53;
  }
  if ( v61 > 0x40 && v60 )
  {
    v51 = v11;
    v54 = v43;
    j_j___libc_free_0_0(v60);
    v11 = v51;
    v43 = v54;
  }
  if ( v43 )
  {
LABEL_13:
    v56 = 0;
    v25 = v11;
  }
  else
  {
LABEL_37:
    if ( v34 && v23 )
    {
      v35 = *(_DWORD *)(a7 + 24);
      if ( v35 <= 0x40 )
        v36 = *(_QWORD *)(a7 + 16) == 0;
      else
        v36 = v35 == (unsigned int)sub_C444A0(a7 + 16);
      v56 = !v36;
    }
    else
    {
      v25 = 0;
      v56 = v55 && v57;
      if ( v56 )
      {
        v45 = *((_DWORD *)a6 + 6);
        if ( v45 <= 0x40 )
          v46 = a6[2] == 0;
        else
          v46 = v45 == (unsigned int)sub_C444A0(a6 + 2);
        v25 = 0;
        v56 = !v46;
      }
    }
  }
  v26 = 0;
LABEL_15:
  sub_C787D0(&v60, a6, a7, v26);
  if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
    j_j___libc_free_0_0(*a6);
  v27 = *((_DWORD *)a6 + 6) <= 0x40u;
  *a6 = v60;
  v28 = v61;
  v61 = 0;
  *((_DWORD *)a6 + 2) = v28;
  if ( v27 || (v29 = a6[2]) == 0 )
  {
    a6[2] = v62;
    result = v63;
    *((_DWORD *)a6 + 6) = v63;
  }
  else
  {
    j_j___libc_free_0_0(v29);
    v27 = v61 <= 0x40;
    a6[2] = v62;
    result = v63;
    *((_DWORD *)a6 + 6) = v63;
    if ( !v27 && v60 )
      result = j_j___libc_free_0_0(v60);
  }
  if ( v25 )
  {
    v31 = *((_DWORD *)a6 + 6);
    result = a6[2];
    if ( v31 > 0x40 )
      result = *(_QWORD *)(result + 8LL * ((v31 - 1) >> 6));
    if ( (result & (1LL << ((unsigned __int8)v31 - 1))) == 0 )
    {
      v37 = *((_DWORD *)a6 + 2);
      v38 = *a6;
      v39 = v37 - 1;
      result = 1LL << ((unsigned __int8)v37 - 1);
      if ( v37 <= 0x40 )
      {
        result |= v38;
        *a6 = result;
        return result;
      }
      goto LABEL_43;
    }
  }
  if ( v56 )
  {
    v32 = *((_DWORD *)a6 + 2);
    v33 = *a6;
    result = 1LL << ((unsigned __int8)v32 - 1);
    if ( v32 > 0x40 )
      v33 = *(_QWORD *)(v33 + 8LL * ((v32 - 1) >> 6));
    if ( (v33 & result) == 0 )
    {
      v40 = *((_DWORD *)a6 + 6);
      v38 = a6[2];
      v39 = v40 - 1;
      result = 1LL << ((unsigned __int8)v40 - 1);
      if ( v40 <= 0x40 )
      {
        result |= v38;
        a6[2] = result;
        return result;
      }
LABEL_43:
      *(_QWORD *)(v38 + 8LL * (v39 >> 6)) |= result;
    }
  }
  return result;
}
