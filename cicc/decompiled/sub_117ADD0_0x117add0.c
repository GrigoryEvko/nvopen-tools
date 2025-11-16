// Function: sub_117ADD0
// Address: 0x117add0
//
unsigned __int8 *__fastcall sub_117ADD0(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  unsigned __int8 *v4; // r12
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // r14
  unsigned __int8 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v34; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+18h] [rbp-A8h] BYREF
  char v36[32]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v37; // [rsp+40h] [rbp-80h]
  const char *v38; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v39[3]; // [rsp+58h] [rbp-68h] BYREF
  __int64 *v40; // [rsp+70h] [rbp-50h]
  __int64 *v41; // [rsp+78h] [rbp-48h]
  __int64 *v42; // [rsp+80h] [rbp-40h]

  v38 = (const char *)&v32;
  v3 = *(_BYTE *)a1 == 86;
  v39[0] = &v33;
  v39[1] = &v34;
  v39[2] = &v35;
  v40 = &v33;
  v41 = &v35;
  v42 = &v34;
  if ( !v3 )
    return 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) == 0 )
  {
    v6 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    v7 = *v6;
    if ( *v6 )
      goto LABEL_6;
    return 0;
  }
  v6 = *(__int64 **)(a1 - 8);
  v7 = *v6;
  if ( !*v6 )
    return 0;
LABEL_6:
  v8 = v6[4];
  v32 = v7;
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) || *(_BYTE *)v8 != 86 || !(unsigned __int8)sub_1178D30(v39, v8) )
    return 0;
  v11 = (*(_BYTE *)(v10 + 7) & 0x40) != 0 ? *(_QWORD *)(v10 - 8) : v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
  v12 = *(_QWORD *)(v11 + 64);
  v13 = *(_QWORD *)(v12 + 16);
  if ( !v13 )
    return 0;
  v4 = *(unsigned __int8 **)(v13 + 8);
  if ( v4 )
    return 0;
  if ( *(_BYTE *)v12 != 86 )
    return 0;
  v14 = (_QWORD *)sub_986520(v12);
  if ( *v14 != *v40 || v14[4] != *v41 || v14[8] != *v42 )
    return 0;
  v15 = v32;
  v16 = v33;
  if ( *(_QWORD *)(v33 + 8) == *(_QWORD *)(v32 + 8) )
  {
    v17 = a2[10];
    v37 = 257;
    v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v17 + 16LL))(v17, 30, v33, v32);
    if ( !v18 )
    {
      LOWORD(v40) = 257;
      v18 = sub_B504D0(30, v16, v15, (__int64)&v38, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
        a2[11],
        v18,
        v36,
        a2[7],
        a2[8]);
      v28 = *a2;
      v29 = *a2 + 16LL * *((unsigned int *)a2 + 2);
      while ( v29 != v28 )
      {
        v30 = *(_QWORD *)(v28 + 8);
        v31 = *(_DWORD *)v28;
        v28 += 16;
        sub_B99FD0(v18, v31, v30);
      }
    }
    v19 = v34;
    LOWORD(v40) = 257;
    v20 = v35;
    v21 = (unsigned __int8 *)sub_BD2C40(72, 3u);
    v4 = v21;
    if ( v21 )
    {
      sub_B44260((__int64)v21, *(_QWORD *)(v20 + 8), 57, 3u, 0, 0);
      if ( *((_QWORD *)v4 - 12) )
      {
        v22 = *((_QWORD *)v4 - 11);
        **((_QWORD **)v4 - 10) = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = *((_QWORD *)v4 - 10);
      }
      *((_QWORD *)v4 - 12) = v18;
      if ( v18 )
      {
        v23 = *(_QWORD *)(v18 + 16);
        *((_QWORD *)v4 - 11) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = v4 - 88;
        *((_QWORD *)v4 - 10) = v18 + 16;
        *(_QWORD *)(v18 + 16) = v4 - 96;
      }
      if ( *((_QWORD *)v4 - 8) )
      {
        v24 = *((_QWORD *)v4 - 7);
        **((_QWORD **)v4 - 6) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = *((_QWORD *)v4 - 6);
      }
      *((_QWORD *)v4 - 8) = v20;
      v25 = *(_QWORD *)(v20 + 16);
      *((_QWORD *)v4 - 7) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = v4 - 56;
      *((_QWORD *)v4 - 6) = v20 + 16;
      *(_QWORD *)(v20 + 16) = v4 - 64;
      if ( *((_QWORD *)v4 - 4) )
      {
        v26 = *((_QWORD *)v4 - 3);
        **((_QWORD **)v4 - 2) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = *((_QWORD *)v4 - 2);
      }
      *((_QWORD *)v4 - 4) = v19;
      if ( v19 )
      {
        v27 = *(_QWORD *)(v19 + 16);
        *((_QWORD *)v4 - 3) = v27;
        if ( v27 )
          *(_QWORD *)(v27 + 16) = v4 - 24;
        *((_QWORD *)v4 - 2) = v19 + 16;
        *(_QWORD *)(v19 + 16) = v4 - 32;
      }
      sub_BD6B50(v4, &v38);
    }
  }
  return v4;
}
