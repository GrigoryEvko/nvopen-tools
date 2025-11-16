// Function: sub_19D71F0
// Address: 0x19d71f0
//
void __fastcall sub_19D71F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rdi
  __int64 v5; // r12
  const char *v6; // rax
  __int64 v7; // rdi
  const char *v8; // r14
  size_t v9; // rdx
  size_t v10; // r13
  size_t v11; // rdx
  const char *v12; // rsi
  size_t v13; // rcx
  int v14; // eax
  __int64 v15; // r13
  int v16; // eax
  int v17; // eax
  __int64 v18; // r14
  unsigned int i; // ecx
  __int64 v20; // rax
  __int64 v21; // rdi
  bool v22; // cc
  int v23; // eax
  __int64 v24; // rdi
  int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rsi
  unsigned __int64 v30; // rcx
  unsigned int v31; // eax
  __int64 v32; // [rsp+0h] [rbp-90h]
  int v33; // [rsp+8h] [rbp-88h]
  int v34; // [rsp+Ch] [rbp-84h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  __int64 v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+20h] [rbp-70h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  int v39; // [rsp+30h] [rbp-60h]
  char v40; // [rsp+37h] [rbp-59h]
  __int64 v41; // [rsp+38h] [rbp-58h]
  __int64 v42; // [rsp+40h] [rbp-50h]
  size_t v43; // [rsp+48h] [rbp-48h]
  __int64 v44; // [rsp+48h] [rbp-48h]
  size_t v45; // [rsp+48h] [rbp-48h]
  __int64 v47; // [rsp+58h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 104 )
  {
    v3 = a1 + 208;
    do
    {
      v4 = *(_QWORD *)(v3 - 72);
      v47 = v3;
      v5 = v3 - 104;
      if ( v4 )
        v4 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
      v6 = sub_1649960(v4);
      v7 = *(_QWORD *)(a1 + 32);
      v8 = v6;
      v10 = v9;
      if ( v7 )
        v7 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
      v12 = sub_1649960(v7);
      v13 = v11;
      if ( v10 > v11 )
      {
        if ( !v11 )
          goto LABEL_44;
        v45 = v11;
        v14 = memcmp(v8, v12, v11);
        v13 = v45;
        if ( !v14 )
        {
LABEL_12:
          if ( v10 < v13 )
            goto LABEL_13;
          goto LABEL_44;
        }
      }
      else if ( !v10 || (v43 = v11, v14 = memcmp(v8, v12, v10), v13 = v43, !v14) )
      {
        if ( v10 != v13 )
          goto LABEL_12;
        v15 = *(_QWORD *)(v3 - 72);
        v28 = *(_QWORD *)(a1 + 32);
        if ( v15 )
        {
          v29 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
          if ( v28 )
          {
            if ( v29 != *(_QWORD *)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF)) )
              goto LABEL_40;
LABEL_46:
            v31 = (unsigned int)sub_16AEA10(v3 - 56, a1 + 48) >> 31;
LABEL_43:
            if ( (_BYTE)v31 )
              goto LABEL_14;
            goto LABEL_44;
          }
          if ( !v29 )
            goto LABEL_46;
LABEL_40:
          v30 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
          if ( !v28 )
            goto LABEL_44;
          v31 = *(_DWORD *)(v28 + 20);
        }
        else
        {
          if ( !v28 )
            goto LABEL_46;
          v31 = *(_DWORD *)(v28 + 20);
          if ( !*(_QWORD *)(v28 - 24LL * (v31 & 0xFFFFFFF)) )
            goto LABEL_46;
          v30 = 0;
        }
        LOBYTE(v31) = *(_QWORD *)(v28 - 24LL * (v31 & 0xFFFFFFF)) > v30;
        goto LABEL_43;
      }
      if ( v14 < 0 )
      {
LABEL_13:
        v15 = *(_QWORD *)(v3 - 72);
LABEL_14:
        v44 = *(_QWORD *)(v3 - 104);
        v42 = *(_QWORD *)(v3 - 96);
        v41 = *(_QWORD *)(v3 - 88);
        v40 = *(_BYTE *)(v3 - 80);
        v38 = *(_QWORD *)(v3 - 64);
        v16 = *(_DWORD *)(v3 - 48);
        *(_DWORD *)(v3 - 48) = 0;
        v39 = v16;
        v37 = *(_QWORD *)(v3 - 56);
        v36 = *(_QWORD *)(v3 - 40);
        v35 = *(_QWORD *)(v3 - 32);
        v17 = *(_DWORD *)(v3 - 16);
        *(_DWORD *)(v3 - 16) = 0;
        v34 = v17;
        v32 = *(_QWORD *)(v3 - 24);
        v33 = *(_DWORD *)(v3 - 8);
        v18 = 0x4EC4EC4EC4EC4EC5LL * ((v5 - a1) >> 3);
        if ( v5 - a1 > 0 )
        {
          for ( i = 0; ; i = *(_DWORD *)(v5 + 56) )
          {
            v20 = *(_QWORD *)(v5 - 104);
            v5 -= 104;
            *(_QWORD *)(v5 + 104) = v20;
            *(_QWORD *)(v5 + 112) = *(_QWORD *)(v5 + 8);
            *(_QWORD *)(v5 + 120) = *(_QWORD *)(v5 + 16);
            *(_BYTE *)(v5 + 128) = *(_BYTE *)(v5 + 24);
            *(_QWORD *)(v5 + 136) = *(_QWORD *)(v5 + 32);
            *(_QWORD *)(v5 + 144) = *(_QWORD *)(v5 + 40);
            if ( i > 0x40 )
            {
              v21 = *(_QWORD *)(v5 + 152);
              if ( v21 )
                j_j___libc_free_0_0(v21);
            }
            v22 = *(_DWORD *)(v5 + 192) <= 0x40u;
            *(_QWORD *)(v5 + 152) = *(_QWORD *)(v5 + 48);
            v23 = *(_DWORD *)(v5 + 56);
            *(_DWORD *)(v5 + 56) = 0;
            *(_DWORD *)(v5 + 160) = v23;
            *(_QWORD *)(v5 + 168) = *(_QWORD *)(v5 + 64);
            *(_QWORD *)(v5 + 176) = *(_QWORD *)(v5 + 72);
            if ( !v22 )
            {
              v24 = *(_QWORD *)(v5 + 184);
              if ( v24 )
                j_j___libc_free_0_0(v24);
            }
            *(_QWORD *)(v5 + 184) = *(_QWORD *)(v5 + 80);
            v25 = *(_DWORD *)(v5 + 88);
            *(_DWORD *)(v5 + 88) = 0;
            *(_DWORD *)(v5 + 192) = v25;
            *(_DWORD *)(v5 + 200) = *(_DWORD *)(v5 + 96);
            if ( !--v18 )
              break;
          }
        }
        v22 = *(_DWORD *)(a1 + 56) <= 0x40u;
        *(_QWORD *)(a1 + 32) = v15;
        *(_QWORD *)a1 = v44;
        *(_QWORD *)(a1 + 8) = v42;
        *(_QWORD *)(a1 + 16) = v41;
        *(_BYTE *)(a1 + 24) = v40;
        *(_QWORD *)(a1 + 40) = v38;
        if ( !v22 )
        {
          v26 = *(_QWORD *)(a1 + 48);
          if ( v26 )
            j_j___libc_free_0_0(v26);
        }
        v22 = *(_DWORD *)(a1 + 88) <= 0x40u;
        *(_QWORD *)(a1 + 48) = v37;
        *(_DWORD *)(a1 + 56) = v39;
        *(_QWORD *)(a1 + 64) = v36;
        *(_QWORD *)(a1 + 72) = v35;
        if ( !v22 )
        {
          v27 = *(_QWORD *)(a1 + 80);
          if ( v27 )
            j_j___libc_free_0_0(v27);
        }
        *(_QWORD *)(a1 + 80) = v32;
        *(_DWORD *)(a1 + 88) = v34;
        *(_DWORD *)(a1 + 96) = v33;
        goto LABEL_31;
      }
LABEL_44:
      sub_19D6DB0((__int64 *)(v3 - 104));
LABEL_31:
      v3 += 104;
    }
    while ( a2 != v47 );
  }
}
