// Function: sub_2DD0F80
// Address: 0x2dd0f80
//
__int64 __fastcall sub_2DD0F80(__int64 a1, _BYTE *a2, size_t a3)
{
  __int64 *v4; // r13
  int v6; // eax
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 result; // rax
  _QWORD *v11; // rdx
  size_t v12; // rax
  _QWORD *v13; // r15
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  _QWORD *v16; // r15
  int v17; // eax
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // r10
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rdx
  _QWORD *v27; // r14
  _QWORD *v28; // rcx
  _QWORD *v29; // r12
  __int64 v30; // rax
  __int64 (__fastcall *v31)(_QWORD *); // rdx
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  unsigned int v34; // r9d
  _QWORD *v35; // r10
  _QWORD *v36; // rcx
  __int64 *v37; // rax
  __int64 *v38; // rax
  _QWORD *v39; // rdi
  size_t v40; // rdx
  unsigned __int64 v41; // r12
  __int64 v42; // rdi
  _QWORD *v43; // [rsp+0h] [rbp-80h]
  _QWORD *v44; // [rsp+8h] [rbp-78h]
  unsigned int v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  _QWORD *v49; // [rsp+20h] [rbp-60h] BYREF
  size_t v50; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v51; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = (__int64 *)(a1 + 200);
  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(a1 + 200), a2, a3, v6);
  if ( v7 != -1 )
  {
    v8 = *(_QWORD *)(a1 + 200);
    v9 = v8 + 8LL * v7;
    if ( v9 != v8 + 8LL * *(unsigned int *)(a1 + 208) )
      return *(_QWORD *)(*(_QWORD *)v9 + 8LL);
  }
  sub_E3FC80((__int64)&v49, a2, a3);
  v51 = src;
  if ( &a2[a3] && !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v50 = a3;
  if ( a3 > 0xF )
  {
    v51 = (_QWORD *)sub_22409D0((__int64)&v51, &v50, 0);
    v39 = v51;
    src[0] = v50;
  }
  else
  {
    if ( a3 == 1 )
    {
      v11 = src;
      LOBYTE(src[0]) = *a2;
      v12 = 1;
      goto LABEL_10;
    }
    if ( !a3 )
    {
      v12 = 0;
      v11 = src;
      goto LABEL_10;
    }
    v39 = src;
  }
  memcpy(v39, a2, a3);
  v12 = v50;
  v11 = v51;
LABEL_10:
  n = v12;
  *((_BYTE *)v11 + v12) = 0;
  v13 = v49;
  v14 = (_QWORD *)v49[1];
  if ( v51 == src )
  {
    v40 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v14 = src[0];
      else
        memcpy(v14, src, n);
      v40 = n;
      v14 = (_QWORD *)v13[1];
    }
    v13[2] = v40;
    *((_BYTE *)v14 + v40) = 0;
    v14 = v51;
  }
  else
  {
    if ( v14 == v49 + 3 )
    {
      v49[1] = v51;
      v13[2] = n;
      v13[3] = src[0];
    }
    else
    {
      v49[1] = v51;
      v15 = v13[3];
      v13[2] = n;
      v13[3] = src[0];
      if ( v14 )
      {
        v51 = v14;
        src[0] = v15;
        goto LABEL_14;
      }
    }
    v51 = src;
    v14 = src;
  }
LABEL_14:
  n = 0;
  *(_BYTE *)v14 = 0;
  if ( v51 != src )
    j_j___libc_free_0((unsigned __int64)v51);
  v16 = v49;
  v17 = sub_C92610();
  v19 = (unsigned int)sub_C92740((__int64)v4, a2, a3, v17);
  v20 = (_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v19);
  v21 = *v20;
  if ( !*v20 )
    goto LABEL_28;
  if ( v21 == -8 )
  {
    --*(_DWORD *)(a1 + 216);
LABEL_28:
    v44 = v20;
    v45 = v19;
    v33 = sub_C7D670(a3 + 17, 8);
    v34 = v45;
    v35 = v44;
    v36 = (_QWORD *)v33;
    if ( a3 )
    {
      v43 = (_QWORD *)v33;
      memcpy((void *)(v33 + 16), a2, a3);
      v34 = v45;
      v35 = v44;
      v36 = v43;
    }
    *((_BYTE *)v36 + a3 + 16) = 0;
    *v36 = a3;
    v36[1] = 0;
    *v35 = v36;
    ++*(_DWORD *)(a1 + 212);
    v37 = (__int64 *)(*(_QWORD *)(a1 + 200) + 8LL * (unsigned int)sub_C929D0(v4, v34));
    v21 = *v37;
    if ( !*v37 || v21 == -8 )
    {
      v38 = v37 + 1;
      do
      {
        do
          v21 = *v38++;
        while ( !v21 );
      }
      while ( v21 == -8 );
    }
  }
  *(_QWORD *)(v21 + 8) = v16;
  v22 = *(unsigned int *)(a1 + 184);
  v23 = *(unsigned int *)(a1 + 188);
  v24 = v22 + 1;
  v25 = *(_DWORD *)(a1 + 184);
  if ( v22 + 1 > v23 )
  {
    v41 = *(_QWORD *)(a1 + 176);
    v42 = a1 + 176;
    if ( v41 > (unsigned __int64)&v49 || (unsigned __int64)&v49 >= v41 + 8 * v22 )
    {
      sub_2DD0E60(v42, v24, v23, v22, v18, v19);
      v22 = *(unsigned int *)(a1 + 184);
      v26 = *(_QWORD *)(a1 + 176);
      v27 = &v49;
      v25 = *(_DWORD *)(a1 + 184);
    }
    else
    {
      sub_2DD0E60(v42, v24, v23, v22, v18, v19);
      v26 = *(_QWORD *)(a1 + 176);
      v22 = *(unsigned int *)(a1 + 184);
      v27 = (_QWORD **)((char *)&v49 + v26 - v41);
      v25 = *(_DWORD *)(a1 + 184);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 176);
    v27 = &v49;
  }
  v28 = (_QWORD *)(v26 + 8 * v22);
  if ( v28 )
  {
    *v28 = *v27;
    *v27 = 0;
    v26 = *(_QWORD *)(a1 + 176);
    v25 = *(_DWORD *)(a1 + 184);
  }
  v29 = v49;
  v30 = (unsigned int)(v25 + 1);
  *(_DWORD *)(a1 + 184) = v30;
  result = *(_QWORD *)(v26 + 8 * v30 - 8);
  if ( v29 )
  {
    v31 = *(__int64 (__fastcall **)(_QWORD *))(*v29 + 8LL);
    if ( v31 == sub_BD9990 )
    {
      v32 = v29[1];
      *v29 = &unk_49DB390;
      if ( (_QWORD *)v32 != v29 + 3 )
      {
        v46 = result;
        j_j___libc_free_0(v32);
        result = v46;
      }
      v47 = result;
      j_j___libc_free_0((unsigned __int64)v29);
      return v47;
    }
    else
    {
      v48 = result;
      ((void (__fastcall *)(_QWORD *, unsigned __int64, __int64 (__fastcall *)(_QWORD *), _QWORD *, __int64, __int64))v31)(
        v29,
        v24,
        v31,
        v28,
        v18,
        v19);
      return v48;
    }
  }
  return result;
}
