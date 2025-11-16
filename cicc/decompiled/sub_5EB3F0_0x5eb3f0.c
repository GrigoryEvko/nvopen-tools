// Function: sub_5EB3F0
// Address: 0x5eb3f0
//
unsigned int *__fastcall sub_5EB3F0(_QWORD *a1)
{
  unsigned int *result; // rax
  __int64 v2; // rsi
  __int64 v4; // rbx
  __int64 v5; // r11
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  unsigned int *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // [rsp-2A8h] [rbp-2A8h]
  __int64 v28; // [rsp-290h] [rbp-290h]
  __int64 *v29; // [rsp-290h] [rbp-290h]
  int v30; // [rsp-284h] [rbp-284h] BYREF
  __int64 v31; // [rsp-280h] [rbp-280h] BYREF
  _QWORD v32[11]; // [rsp-278h] [rbp-278h] BYREF
  __int64 v33; // [rsp-220h] [rbp-220h] BYREF
  _QWORD v34[59]; // [rsp-218h] [rbp-218h] BYREF
  __int64 v35; // [rsp-40h] [rbp-40h] BYREF

  result = &dword_4F077BC;
  v2 = dword_4F077BC;
  if ( !dword_4F077BC )
    return result;
  if ( (*((_BYTE *)a1 + 170) & 0x10) == 0 )
    return result;
  v4 = *a1;
  result = *(unsigned int **)(*a1 + 104LL);
  if ( !result )
    return result;
  v5 = *(_QWORD *)(v4 + 64);
  v6 = *((_QWORD *)result + 1);
  *((_QWORD *)result + 1) = 0;
  if ( *((char *)a1 + 170) < 0 )
    return result;
  v7 = *(_QWORD *)(v4 + 96);
  if ( !v7 )
  {
    if ( !v6 )
      return result;
    v29 = *(__int64 **)(unk_4F04C68 + 776LL * dword_4F04C64 + 336);
    goto LABEL_11;
  }
  v7 = *(_QWORD *)(v7 + 32);
  v28 = v5;
  result = (unsigned int *)sub_8933F0(v7, v4, &dword_4F063F8);
  v5 = v28;
  if ( (_DWORD)result )
  {
    *((_BYTE *)a1 + 177) = 1;
    result = (unsigned int *)sub_72C9A0();
    a1[23] = result;
    return result;
  }
  if ( v6 )
  {
    v29 = *(__int64 **)(unk_4F04C68 + 776LL * dword_4F04C64 + 336);
    if ( v7 )
    {
      v27 = v5;
      sub_893490(v7);
      v5 = v27;
    }
    v2 = dword_4F077BC;
LABEL_11:
    sub_866000(v5, v2, 1);
    sub_7B8190();
    sub_7BC160(v6);
    memset(v34, 0, sizeof(v34));
    v34[19] = v34;
    v34[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v34[22]) |= 1u;
    BYTE1(v34[15]) |= 0x40u;
    v10 = a1[15];
    v34[0] = v4;
    v34[35] = v10;
    v34[36] = v10;
    if ( (*((_BYTE *)a1 + 174) & 0x40) != 0 )
    {
      HIBYTE(v34[15]) |= 8u;
      LOBYTE(v34[22]) |= 1u;
    }
    if ( unk_4F077C4 == 2 && unk_4F07778 > 201702
      || dword_4F077BC
      && (v11 = a1[27]) != 0
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v11 + 16) + 96LL) + 56LL) + 168LL) & 1) != 0 )
    {
      v30 = 0;
      memset(v32, 0, sizeof(v32));
      v12 = 0;
      v32[8] = *(_QWORD *)&dword_4F063F8;
      if ( word_4F06418[0] == 27 )
      {
        v12 = 0;
        if ( (a1[22] & 1) == 0 )
        {
          sub_7B8B50(&v33, v2, dword_4F077BC, 0);
          v12 = 1;
        }
      }
      BYTE4(v34[15]) &= ~0x80u;
      v13 = v34;
      v14 = (unsigned int *)&v31;
      v31 = a1[8];
      sub_638AC0(v34, &v31, 2, v12, &v30);
      BYTE4(v34[15]) &= ~0x80u;
      if ( *((_BYTE *)a1 + 136) == 1 && (*((_BYTE *)a1 + 172) & 0x20) != 0 )
        *((_BYTE *)a1 + 136) = 0;
    }
    else
    {
      v14 = (unsigned int *)sub_724DC0(&v35, v2, dword_4F077BC, 0, v8, v9);
      v32[0] = v14;
      a1[24] = *(_QWORD *)&dword_4F063F8;
      sub_6D6DB0(v34, v14);
      *((_BYTE *)a1 + 177) = 1;
      a1[23] = sub_724E50(v32, v14, v23, v24, v25, v26);
      a1[25] = unk_4F061D8;
      sub_649FB0(v34);
      if ( word_4F06418[0] != 9 )
      {
        v14 = &dword_4F063F8;
        sub_6851C0(65, &dword_4F063F8);
      }
      v13 = (_QWORD *)a1[15];
      if ( (unsigned int)sub_8D23E0(v13) )
      {
        v13 = a1;
        sub_631D50(a1);
      }
    }
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(v13, v14, v15, v16);
    sub_7B8B50(v13, v14, v15, v16);
    sub_7B8260();
    sub_866010(v13, v14, v17, v18, v19);
    if ( v7 )
      sub_893500(v7);
    result = (unsigned int *)(unk_4F04C68 + 776LL * dword_4F04C64);
    if ( *((__int64 **)result + 42) != v29 )
    {
      v20 = a1[12];
      v21 = *v29;
      v22 = *(__int64 **)(v20 + 8);
      *v22 = *v29;
      *(_QWORD *)(v21 + 8) = v22;
      **(_QWORD **)(unk_4F04C68 + 776LL * dword_4F04C64 + 336) = v20;
      result = (unsigned int *)(unk_4F04C68 + 776LL * dword_4F04C64);
      *(_QWORD *)(v20 + 8) = *((_QWORD *)result + 42);
      *((_QWORD *)result + 42) = v20;
      *v29 = 0;
    }
  }
  return result;
}
