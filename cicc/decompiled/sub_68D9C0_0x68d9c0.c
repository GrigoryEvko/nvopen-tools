// Function: sub_68D9C0
// Address: 0x68d9c0
//
__int64 __fastcall sub_68D9C0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 *a5, __int64 a6, char a7)
{
  bool v7; // r15
  _QWORD *v10; // rbx
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rsi
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // r11d
  __int64 v18; // r9
  __int64 **v19; // rdx
  char v20; // cl
  __int64 v21; // r9
  __int64 v22; // r10
  __int64 v23; // rax
  __int64 v24; // r9
  int v26; // eax
  int v27; // eax
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 v33; // rax
  char i; // dl
  int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // rax
  __int64 v40; // r9
  int v41; // eax
  int v42; // eax
  __int64 **v44; // [rsp+18h] [rbp-248h]
  bool v45; // [rsp+20h] [rbp-240h]
  __int64 v46; // [rsp+20h] [rbp-240h]
  int v47; // [rsp+28h] [rbp-238h]
  __int64 v48; // [rsp+28h] [rbp-238h]
  __int64 v49; // [rsp+28h] [rbp-238h]
  __int64 v50; // [rsp+28h] [rbp-238h]
  __int64 v51; // [rsp+28h] [rbp-238h]
  __int64 v53; // [rsp+30h] [rbp-230h]
  __int64 v54; // [rsp+30h] [rbp-230h]
  __int64 v55; // [rsp+30h] [rbp-230h]
  __int64 v56; // [rsp+30h] [rbp-230h]
  __int64 v57; // [rsp+30h] [rbp-230h]
  __int64 v58; // [rsp+30h] [rbp-230h]
  __int64 v60; // [rsp+48h] [rbp-218h] BYREF
  _QWORD v61[66]; // [rsp+50h] [rbp-210h] BYREF

  v7 = 1;
  v10 = (_QWORD *)a1;
  v11 = *(_QWORD *)a1;
  v12 = *(unsigned __int8 *)(unk_4D03C50 + 16LL);
  if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 4) == 0 )
    v7 = (unsigned __int8)v12 <= 3u;
  v60 = 0;
  if ( (_BYTE)v12 == 1 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, v12, a4, a5, v11) )
    {
      a1 = 976;
      sub_6851C0(0x3D0u, a3);
    }
    goto LABEL_9;
  }
  v13 = dword_4D047EC;
  if ( dword_4D047EC )
  {
    a1 = v11;
    v51 = v11;
    v28 = sub_8D4070(v11);
    v11 = v51;
    if ( v28 )
    {
      if ( (unsigned int)sub_6E5430(a1, v13, v29, v30, v31, v51) )
      {
        a1 = 975;
        sub_6851C0(0x3CFu, a3);
      }
LABEL_9:
      v16 = sub_72C930(a1);
      v17 = 1;
      v18 = v16;
      goto LABEL_10;
    }
  }
  v14 = *(_BYTE *)(v11 + 140);
  if ( v14 == 12 )
  {
    v15 = v11;
    do
    {
      v15 = *(_QWORD *)(v15 + 160);
      v14 = *(_BYTE *)(v15 + 140);
    }
    while ( v14 == 12 );
  }
  if ( !v14 )
    goto LABEL_9;
  v49 = v11;
  v26 = sub_8D25A0(v11);
  v18 = v49;
  if ( !v26 )
  {
    a1 = v49;
    v35 = sub_8D3410(v49);
    v18 = v49;
    if ( !v35 )
      goto LABEL_57;
    v39 = sub_8D4050(v49);
    v40 = v49;
    a1 = v39;
    if ( dword_4F077C4 == 2 )
    {
      v42 = sub_8D23B0(v39);
      v40 = v49;
      if ( v42 )
      {
        sub_8AE000(a1);
        v40 = v49;
      }
    }
    v49 = v40;
    v41 = sub_8D25A0(a1);
    v18 = v49;
    if ( !v41 )
    {
LABEL_57:
      if ( (unsigned int)sub_6E5430(a1, v13, v36, v37, v38, v18) )
      {
        a1 = 977;
        sub_685360(0x3D1u, a3, v49);
      }
      goto LABEL_9;
    }
  }
  v17 = HIDWORD(qword_4F077B4);
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( dword_4F077C4 == 2 || (v17 = 0, unk_4F07778 <= 199900) )
    {
      v50 = v18;
      v27 = sub_6E53E0(5, 1607, dword_4F07508);
      v18 = v50;
      v17 = v27;
      if ( v27 )
      {
        v17 = unk_4D04320;
        if ( unk_4D04320 )
        {
          sub_684B30(0x647u, dword_4F07508);
          v18 = v50;
          v17 = 0;
        }
      }
    }
  }
LABEL_10:
  v19 = (__int64 **)&v60;
  v20 = *(_BYTE *)(unk_4D03C50 + 19LL);
  if ( (v20 & 2) == 0 )
    v19 = 0;
  *(_BYTE *)(unk_4D03C50 + 19LL) = v20 | 8;
  memset(v61, 0, 0x1D8u);
  v61[19] = v61;
  v61[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v61[22]) |= 1u;
  v45 = (v20 & 8) != 0;
  v47 = v17;
  v61[36] = v18;
  v44 = v19;
  sub_6E6990(&v61[17]);
  LOBYTE(v61[22]) = (2 * v7) | v61[22] & 0xFD;
  sub_638970((__int64)v61, a5, v44);
  v21 = v61[36];
  v22 = v61[18];
  *v10 = v61[36];
  *(_BYTE *)(unk_4D03C50 + 19LL) = *(_BYTE *)(unk_4D03C50 + 19LL) & 0xF7 | (8 * v45);
  if ( v22 )
  {
    if ( v60 && !*(_QWORD *)(v22 + 120) )
    {
      v46 = v21;
      v55 = v22;
      sub_6EA060(v22);
      v21 = v46;
      v22 = v55;
      if ( !v47 )
        goto LABEL_19;
    }
    else if ( !v47 )
    {
LABEL_19:
      if ( v7 && *(_BYTE *)(v22 + 48) == 2 )
      {
        v32 = *(_QWORD *)(v22 + 56);
        v33 = *(_QWORD *)(v32 + 128);
        for ( i = *(_BYTE *)(v33 + 140); i == 12; i = *(_BYTE *)(v33 + 140) )
          v33 = *(_QWORD *)(v33 + 160);
        if ( i )
        {
          if ( HIDWORD(qword_4F077B4) && (a7 & 4) == 0 )
          {
            v58 = v21;
            sub_6E6A50(v32, a6);
            *(_BYTE *)(a6 + 17) = 2;
            v24 = v58;
          }
          else
          {
            v56 = v21;
            sub_6F8FA0(v32, a6);
            v24 = v56;
          }
        }
        else
        {
          v57 = v21;
          sub_6E6260(a6);
          v24 = v57;
        }
      }
      else
      {
        v53 = v21;
        v48 = v22;
        v23 = sub_6EC670(v21, v22, 1, 0);
        *(_BYTE *)(v48 + 49) &= ~0x10u;
        sub_6E7150(v23, a6);
        v24 = v53;
      }
      goto LABEL_22;
    }
  }
  v54 = v21;
  sub_6E6260(a6);
  v24 = v54;
  if ( a4 )
    *(_BYTE *)(a4 + 56) = 1;
LABEL_22:
  sub_6E41D0(a6, 0, 1, a2, a3, v24);
  return sub_6E26D0(1, a6);
}
