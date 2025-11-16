// Function: sub_1BA1A20
// Address: 0x1ba1a20
//
void __fastcall sub_1BA1A20(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  unsigned int *v4; // r8
  int v5; // r9d
  unsigned __int64 v6; // rbx
  _BYTE *v7; // rdi
  int v8; // eax
  unsigned int v9; // eax
  __int64 v10; // r13
  unsigned int i; // r14d
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 *v18; // rax
  bool v19; // cc
  __int64 *v20; // rax
  _QWORD *v21; // rbx
  unsigned int v22; // ebx
  __int64 v23; // rcx
  unsigned int v24; // edx
  _QWORD *v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  unsigned __int64 *v36; // r12
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned __int8 *v42; // rsi
  unsigned int v43; // [rsp+14h] [rbp-DCh]
  __int64 v44; // [rsp+20h] [rbp-D0h]
  __int64 v45; // [rsp+28h] [rbp-C8h]
  __int64 v46; // [rsp+30h] [rbp-C0h]
  _QWORD *v47; // [rsp+30h] [rbp-C0h]
  __int64 v48; // [rsp+38h] [rbp-B8h]
  __int64 v49; // [rsp+38h] [rbp-B8h]
  __int64 *v50; // [rsp+40h] [rbp-B0h]
  __int64 *v51; // [rsp+48h] [rbp-A8h]
  __int64 v52; // [rsp+58h] [rbp-98h]
  const char *v53; // [rsp+60h] [rbp-90h] BYREF
  char v54; // [rsp+70h] [rbp-80h]
  char v55; // [rsp+71h] [rbp-7Fh]
  __int64 v56[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v57; // [rsp+90h] [rbp-60h]
  _BYTE *v58; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-48h]
  _BYTE s[64]; // [rsp+B0h] [rbp-40h] BYREF

  v3 = a1;
  sub_1B91520(*(_QWORD *)(a2 + 224), *(__int64 **)(a2 + 176), *(_QWORD *)(a1 + 40));
  v6 = *(unsigned int *)(a2 + 4);
  v7 = s;
  v8 = *(_DWORD *)(*(_QWORD *)(v3 + 40) + 20LL);
  v58 = s;
  v43 = v8 & 0xFFFFFFF;
  v59 = 0x200000000LL;
  if ( (unsigned int)v6 > 2 )
  {
    sub_16CD150((__int64)&v58, s, v6, 8, (int)&v58, v5);
    v7 = v58;
  }
  LODWORD(v59) = v6;
  if ( 8 * v6 )
    memset(v7, 0, 8 * v6);
  v9 = *(_DWORD *)(a2 + 4);
  if ( v43 )
  {
    v52 = 0;
    v10 = v3;
    do
    {
      if ( v9 )
      {
        for ( i = 0; i < v9; ++i )
        {
          while ( 1 )
          {
            v15 = *(_QWORD *)(v10 + 40);
            v12 = (*(_BYTE *)(v15 + 23) & 0x40) != 0
                ? *(_QWORD *)(v15 - 8)
                : v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
            v13 = sub_1B9C240(*(unsigned int **)(a2 + 224), *(__int64 **)(v12 + 24 * v52), i);
            if ( (_DWORD)v52 )
              break;
            v14 = i++;
            *(_QWORD *)&v58[8 * v14] = v13;
            v9 = *(_DWORD *)(a2 + 4);
            if ( v9 <= i )
              goto LABEL_19;
          }
          v16 = sub_1BA16F0(a2, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 48) + 40LL) + 8 * v52), i);
          v55 = 1;
          v17 = v16;
          v18 = *(__int64 **)(a2 + 176);
          v54 = 3;
          v19 = *(_BYTE *)(v17 + 16) <= 0x10u;
          v51 = v18;
          v53 = "predphi";
          v20 = (__int64 *)&v58[8 * i];
          v50 = v20;
          if ( v19 && *(_BYTE *)(v13 + 16) <= 0x10u && *(_BYTE *)(*v20 + 16) <= 0x10u )
          {
            v21 = (_QWORD *)sub_15A2DC0(v17, (__int64 *)v13, *v20, 0);
          }
          else
          {
            v46 = v17;
            v48 = *v20;
            v57 = 257;
            v25 = sub_1648A60(56, 3u);
            v21 = v25;
            if ( v25 )
            {
              v44 = v46;
              v45 = v48;
              v47 = v25 - 9;
              v49 = (__int64)v25;
              sub_15F1EA0((__int64)v25, *(_QWORD *)v13, 55, (__int64)(v25 - 9), 3, 0);
              if ( *(v21 - 9) )
              {
                v26 = *(v21 - 8);
                v27 = *(v21 - 7) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v27 = v26;
                if ( v26 )
                  *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
              }
              *(v21 - 9) = v44;
              v28 = *(_QWORD *)(v44 + 8);
              *(v21 - 8) = v28;
              if ( v28 )
                *(_QWORD *)(v28 + 16) = (unsigned __int64)(v21 - 8) | *(_QWORD *)(v28 + 16) & 3LL;
              *(v21 - 7) = (v44 + 8) | *(v21 - 7) & 3LL;
              *(_QWORD *)(v44 + 8) = v47;
              if ( *(v21 - 6) )
              {
                v29 = *(v21 - 5);
                v30 = *(v21 - 4) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v30 = v29;
                if ( v29 )
                  *(_QWORD *)(v29 + 16) = *(_QWORD *)(v29 + 16) & 3LL | v30;
              }
              *(v21 - 6) = v13;
              v31 = *(_QWORD *)(v13 + 8);
              *(v21 - 5) = v31;
              if ( v31 )
                *(_QWORD *)(v31 + 16) = (unsigned __int64)(v21 - 5) | *(_QWORD *)(v31 + 16) & 3LL;
              *(v21 - 4) = (v13 + 8) | *(v21 - 4) & 3LL;
              *(_QWORD *)(v13 + 8) = v21 - 6;
              if ( *(v21 - 3) )
              {
                v32 = *(v21 - 2);
                v33 = *(v21 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v33 = v32;
                if ( v32 )
                  *(_QWORD *)(v32 + 16) = *(_QWORD *)(v32 + 16) & 3LL | v33;
              }
              *(v21 - 3) = v45;
              if ( v45 )
              {
                v34 = *(_QWORD *)(v45 + 8);
                *(v21 - 2) = v34;
                if ( v34 )
                  *(_QWORD *)(v34 + 16) = (unsigned __int64)(v21 - 2) | *(_QWORD *)(v34 + 16) & 3LL;
                *(v21 - 1) = (v45 + 8) | *(v21 - 1) & 3LL;
                *(_QWORD *)(v45 + 8) = v21 - 3;
              }
              sub_164B780((__int64)v21, v56);
            }
            else
            {
              v49 = 0;
            }
            v35 = v51[1];
            if ( v35 )
            {
              v36 = (unsigned __int64 *)v51[2];
              sub_157E9D0(v35 + 40, (__int64)v21);
              v37 = v21[3];
              v38 = *v36;
              v21[4] = v36;
              v38 &= 0xFFFFFFFFFFFFFFF8LL;
              v21[3] = v38 | v37 & 7;
              *(_QWORD *)(v38 + 8) = v21 + 3;
              *v36 = *v36 & 7 | (unsigned __int64)(v21 + 3);
            }
            sub_164B780(v49, (__int64 *)&v53);
            v39 = *v51;
            if ( *v51 )
            {
              v56[0] = *v51;
              sub_1623A60((__int64)v56, v39, 2);
              v40 = v21[6];
              v41 = (__int64)(v21 + 6);
              if ( v40 )
              {
                sub_161E7C0((__int64)(v21 + 6), v40);
                v41 = (__int64)(v21 + 6);
              }
              v42 = (unsigned __int8 *)v56[0];
              v21[6] = v56[0];
              if ( v42 )
                sub_1623210((__int64)v56, v42, v41);
            }
          }
          *v50 = (__int64)v21;
          v9 = *(_DWORD *)(a2 + 4);
        }
      }
      else if ( v43 <= (unsigned int)++v52 )
      {
        goto LABEL_24;
      }
LABEL_19:
      ++v52;
    }
    while ( v43 > (unsigned int)v52 );
    v3 = v10;
  }
  if ( v9 )
  {
    v22 = 0;
    do
    {
      v23 = *(_QWORD *)&v58[8 * v22];
      v24 = v22++;
      sub_1B99BD0(*(unsigned int **)(a2 + 184), *(_QWORD *)(v3 + 40), v24, v23, v4, v5);
    }
    while ( *(_DWORD *)(a2 + 4) > v22 );
  }
LABEL_24:
  if ( v58 != s )
    _libc_free((unsigned __int64)v58);
}
