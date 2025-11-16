// Function: sub_32200A0
// Address: 0x32200a0
//
void __fastcall sub_32200A0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v6; // r15
  int *v7; // rbx
  __int64 v8; // rax
  int *v9; // rdx
  signed __int64 v10; // rax
  int *v11; // rax
  int v12; // eax
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rax
  void *v21; // r12
  __int64 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-68h] BYREF
  __int64 v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  unsigned __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]
  __int64 v30; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD *)a3;
  v24 = a2;
  v25 = 0;
  v26 = 0;
  if ( v6 )
  {
    v25 = *(_QWORD *)(v6 + 16);
    v26 = *(_QWORD *)(v6 + 24);
    sub_3243D60(a4, v6);
    if ( sub_AF46F0(v6) )
    {
      v18 = *(_QWORD *)(a3 + 8);
      v19 = *(_DWORD *)(v18 + 20);
      LOBYTE(v18) = *(_BYTE *)(v18 + 16);
      HIDWORD(v27) = v19;
      LOBYTE(v27) = v18;
      sub_32435C0(a4, &v27, v6);
      sub_3243610(a4, &v25);
      v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 232) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 16LL));
      v15 = HIDWORD(v27);
      v16 = v20;
      goto LABEL_33;
    }
  }
  else
  {
    sub_3243D60(a4, 0);
  }
  v7 = *(int **)(a3 + 8);
  if ( *(_BYTE *)(a3 + 72) )
  {
    v8 = 6LL * *(unsigned int *)(a3 + 16);
    v9 = &v7[v8];
    v10 = 0xAAAAAAAAAAAAAAABLL * ((v8 * 4) >> 3);
    if ( v10 >> 2 )
    {
      v11 = &v7[24 * (v10 >> 2)];
      while ( *v7 || v7[5] )
      {
        if ( !v7[6] && !v7[11] )
        {
          v7 += 6;
          break;
        }
        if ( !v7[12] && !v7[17] )
        {
          v7 += 12;
          break;
        }
        if ( !v7[18] && !v7[23] )
        {
          v7 += 18;
          break;
        }
        v7 += 24;
        if ( v11 == v7 )
        {
          v10 = 0xAAAAAAAAAAAAAAABLL * (((char *)v9 - (char *)v7) >> 3);
          goto LABEL_22;
        }
      }
LABEL_8:
      if ( v9 != v7 )
        return;
      goto LABEL_9;
    }
LABEL_22:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          goto LABEL_9;
        goto LABEL_25;
      }
      if ( !*v7 && !v7[5] )
        goto LABEL_8;
      v7 += 6;
    }
    if ( !*v7 && !v7[5] )
      goto LABEL_8;
    v7 += 6;
LABEL_25:
    if ( !*v7 && !v7[5] )
      goto LABEL_8;
LABEL_9:
    v27 = a4;
    v28 = &v24;
    v29 = a1;
    v30 = a3;
    sub_3244260(a4, &v25, sub_321BD10, &v27);
    return;
  }
  v12 = *v7;
  if ( *v7 == 1 )
  {
    v17 = *((_QWORD *)v7 + 1);
    if ( v24 && (unsigned int)(*(_DWORD *)(v24 + 44) - 5) <= 1 )
      sub_32432C0(a4, v17);
    else
      sub_3243300(a4, v17);
    goto LABEL_34;
  }
  if ( !v12 )
  {
    v13 = v7[5];
    if ( !*((_BYTE *)v7 + 16) )
      *(_BYTE *)(a4 + 100) = *(_BYTE *)(a4 + 100) & 0xF8 | 2;
    v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 232) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 16LL));
    v15 = v13;
    v16 = v14;
LABEL_33:
    if ( (unsigned __int8)sub_3243770(a4, v16, &v25, v15, 0) )
      goto LABEL_34;
    return;
  }
  if ( v12 == 4 )
  {
    sub_3243F80(a4, (unsigned int)v7[4], v7[5]);
    goto LABEL_34;
  }
  if ( v12 != 2 )
  {
LABEL_34:
    sub_3244870(a4, &v25);
    return;
  }
  if ( (unsigned __int16)sub_31DF670(a1) > 3u && *(_DWORD *)(*(_QWORD *)(a1 + 760) + 6224LL) != 3 && v25 == v26 )
  {
    sub_32433D0(a4, *((_QWORD *)v7 + 1) + 24LL);
    goto LABEL_34;
  }
  v21 = sub_C33340();
  v22 = (__int64 *)(*((_QWORD *)v7 + 1) + 24LL);
  if ( (void *)*v22 == v21 )
    sub_C3E660((__int64)&v27, (__int64)v22);
  else
    sub_C3A850((__int64)&v27, v22);
  if ( (unsigned int)v28 <= 0x40 )
  {
    v23 = *((_QWORD *)v7 + 1);
    if ( v21 == *(void **)(v23 + 24) )
      sub_C3E660((__int64)&v27, v23 + 24);
    else
      sub_C3A850((__int64)&v27, (__int64 *)(v23 + 24));
    sub_3243320(a4, &v27);
    if ( (unsigned int)v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    goto LABEL_34;
  }
  if ( v27 )
    j_j___libc_free_0_0(v27);
}
