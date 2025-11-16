// Function: sub_35DEB20
// Address: 0x35deb20
//
_BYTE *__fastcall sub_35DEB20(__int64 *a1, unsigned __int64 a2, __int64 **a3)
{
  __int64 v5; // rdi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // eax
  int v12; // edx
  unsigned int v13; // eax
  __int64 v14; // rsi
  _BYTE *v15; // r13
  _QWORD *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 (__fastcall *v23)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v24; // rdi
  _QWORD *v25; // rax
  __int64 *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // r12
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  int v33; // edi
  char v34[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v35; // [rsp+20h] [rbp-70h]
  char v36[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v37; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)a2 <= 0x1Cu || *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 12 )
    return 0;
  v5 = *a1;
  if ( *(_BYTE *)(v5 + 220) )
  {
    v7 = *(_QWORD **)(v5 + 200);
    v8 = &v7[*(unsigned int *)(v5 + 212)];
    if ( v7 != v8 )
    {
      while ( *v7 != a2 )
      {
        if ( v8 == ++v7 )
          goto LABEL_27;
      }
      goto LABEL_8;
    }
  }
  else
  {
    v26 = sub_C8CA60(v5 + 192, a2);
    v5 = *a1;
    if ( v26 )
      goto LABEL_8;
  }
LABEL_27:
  if ( *(_BYTE *)(v5 + 92) )
  {
    v27 = *(_QWORD **)(v5 + 72);
    v28 = &v27[*(unsigned int *)(v5 + 84)];
    if ( v27 == v28 )
      return 0;
    while ( *v27 != a2 )
    {
      if ( v28 == ++v27 )
        return 0;
    }
  }
  else
  {
    if ( !sub_C8CA60(v5 + 64, a2) )
      return 0;
    v5 = *a1;
  }
LABEL_8:
  v9 = *(_QWORD *)(v5 + 24);
  v10 = *(_QWORD *)(v9 + 8);
  v11 = *(_DWORD *)(v9 + 24);
  if ( v11 )
  {
    v12 = v11 - 1;
    v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = *(_QWORD *)(v10 + 8LL * v13);
    if ( v14 == a2 )
      return 0;
    v33 = 1;
    while ( v14 != -4096 )
    {
      v13 = v12 & (v33 + v13);
      v14 = *(_QWORD *)(v10 + 8LL * v13);
      if ( v14 == a2 )
        return 0;
      ++v33;
    }
  }
  sub_D5F1F0(a1[1], a2);
  v21 = a1[1];
  v35 = 257;
  if ( a3 == *(__int64 ***)(a2 + 8) )
  {
    v15 = (_BYTE *)a2;
    goto LABEL_18;
  }
  v22 = *(_QWORD *)(v21 + 80);
  v23 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v22 + 120LL);
  if ( v23 != sub_920130 )
  {
    v15 = (_BYTE *)v23(v22, 38u, (_BYTE *)a2, (__int64)a3);
    goto LABEL_17;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v15 = (_BYTE *)sub_ADAB70(38, a2, a3, 0);
    else
      v15 = (_BYTE *)sub_AA93C0(0x26u, a2, (__int64)a3);
LABEL_17:
    if ( v15 )
      goto LABEL_18;
  }
  v37 = 257;
  v15 = (_BYTE *)sub_B51D30(38, a2, (__int64)a3, (__int64)v36, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, char *, _QWORD, _QWORD))(**(_QWORD **)(v21 + 88) + 16LL))(
    *(_QWORD *)(v21 + 88),
    v15,
    v34,
    *(_QWORD *)(v21 + 56),
    *(_QWORD *)(v21 + 64));
  v29 = *(_QWORD *)v21;
  v30 = *(_QWORD *)v21 + 16LL * *(unsigned int *)(v21 + 8);
  if ( *(_QWORD *)v21 != v30 )
  {
    do
    {
      v31 = *(_QWORD *)(v29 + 8);
      v32 = *(_DWORD *)v29;
      v29 += 16;
      sub_B99FD0((__int64)v15, v32, v31);
    }
    while ( v30 != v29 );
  }
LABEL_18:
  if ( *v15 <= 0x1Cu )
    return 0;
  v24 = *a1;
  if ( !*(_BYTE *)(*a1 + 92) )
  {
LABEL_25:
    sub_C8CC70(v24 + 64, (__int64)v15, (__int64)v17, v18, v19, v20);
    return v15;
  }
  v25 = *(_QWORD **)(v24 + 72);
  v18 = *(unsigned int *)(v24 + 84);
  v17 = &v25[v18];
  if ( v25 == v17 )
  {
LABEL_23:
    if ( (unsigned int)v18 < *(_DWORD *)(v24 + 80) )
    {
      *(_DWORD *)(v24 + 84) = v18 + 1;
      *v17 = v15;
      ++*(_QWORD *)(v24 + 64);
      return v15;
    }
    goto LABEL_25;
  }
  while ( v15 != (_BYTE *)*v25 )
  {
    if ( v17 == ++v25 )
      goto LABEL_23;
  }
  return v15;
}
