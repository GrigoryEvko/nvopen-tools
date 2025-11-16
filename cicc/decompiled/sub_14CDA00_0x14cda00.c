// Function: sub_14CDA00
// Address: 0x14cda00
//
void __fastcall sub_14CDA00(__int64 a1, __int64 a2)
{
  unsigned int v4; // r12d
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r8
  __int64 v14; // r12
  int v15; // ebx
  unsigned __int64 v16; // r15
  _BYTE *v17; // rbx
  _QWORD *v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  bool v28; // zf
  _BYTE *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rax
  _QWORD *v35; // [rsp+8h] [rbp-288h]
  __int64 v36; // [rsp+10h] [rbp-280h]
  _QWORD *v37; // [rsp+10h] [rbp-280h]
  __int64 v38; // [rsp+10h] [rbp-280h]
  _QWORD *v39; // [rsp+10h] [rbp-280h]
  _QWORD *v40; // [rsp+28h] [rbp-268h] BYREF
  __int64 *v41[2]; // [rsp+30h] [rbp-260h] BYREF
  __int64 v42; // [rsp+40h] [rbp-250h]
  int v43; // [rsp+48h] [rbp-248h]
  _BYTE *v44; // [rsp+50h] [rbp-240h] BYREF
  __int64 v45; // [rsp+58h] [rbp-238h]
  _BYTE v46[560]; // [rsp+60h] [rbp-230h] BYREF

  v4 = 0;
  v5 = *(_BYTE *)(a2 + 23);
  v44 = v46;
  v45 = 0x1000000000LL;
  v40 = &v44;
  while ( 1 )
  {
    if ( v5 >= 0 )
    {
      LODWORD(v11) = 0;
      goto LABEL_10;
    }
    v6 = sub_1648A40(a2);
    v8 = v6 + v7;
    if ( *(char *)(a2 + 23) >= 0 )
      break;
    if ( v4 == (unsigned int)((v8 - sub_1648A40(a2)) >> 4) )
      goto LABEL_12;
    if ( *(char *)(a2 + 23) < 0 )
    {
      v9 = sub_1648A40(a2);
      goto LABEL_6;
    }
LABEL_11:
    v9 = 0;
LABEL_6:
    v5 = *(_BYTE *)(a2 + 23);
    v10 = (_QWORD *)(16LL * v4);
    if ( 3LL * *(unsigned int *)((char *)v10 + v9 + 8) == 3LL * *(unsigned int *)((char *)v10 + v9 + 12) )
      goto LABEL_7;
    if ( v5 >= 0 )
    {
      v34 = (_QWORD *)*v10;
      if ( *(_QWORD *)*v10 != 6 || *((_DWORD *)v34 + 4) != 1869506409 )
        goto LABEL_67;
      if ( *((_WORD *)v34 + 10) != 25970 )
      {
        v33 = 0;
        goto LABEL_65;
      }
    }
    else
    {
      v31 = sub_1648A40(a2);
      v5 = *(_BYTE *)(a2 + 23);
      v32 = *(_QWORD *)(v31 + 16LL * v4);
      if ( *(_QWORD *)v32 != 6 || *(_DWORD *)(v32 + 16) != 1869506409 || *(_WORD *)(v32 + 20) != 25970 )
      {
        if ( v5 >= 0 )
LABEL_67:
          v33 = 0;
        else
          v33 = sub_1648A40(a2);
LABEL_65:
        sub_14CBC40(
          (__int64 *)&v40,
          *(_QWORD *)(a2
                    + 24
                    * (*(unsigned int *)((char *)v10 + v33 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
          v4);
        v5 = *(_BYTE *)(a2 + 23);
      }
    }
LABEL_7:
    ++v4;
  }
  v11 = v8 >> 4;
LABEL_10:
  if ( v4 != (_DWORD)v11 )
    goto LABEL_11;
LABEL_12:
  v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  sub_14CBC40((__int64 *)&v40, v12, -1);
  if ( *(_BYTE *)(v12 + 16) == 75 )
  {
    v13 = *(_QWORD *)(v12 - 48);
    if ( v13 )
    {
      v14 = *(_QWORD *)(v12 - 24);
      if ( v14 )
      {
        v15 = *(unsigned __int16 *)(v12 + 18);
        v36 = v13;
        sub_14CBC40((__int64 *)&v40, v13, -1);
        BYTE1(v15) &= ~0x80u;
        sub_14CBC40((__int64 *)&v40, v14, -1);
        if ( v15 == 32 )
        {
          v41[0] = (__int64 *)&v40;
          sub_14CC460(v41, v36);
          sub_14CC460(v41, v14);
        }
      }
    }
  }
  v16 = (unsigned __int64)v44;
  v17 = &v44[32 * (unsigned int)v45];
  if ( v17 != v44 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v18 = sub_14CCFE0(a1, *(_QWORD *)(v16 + 16));
        v19 = *v18;
        v20 = 32LL * *((unsigned int *)v18 + 2);
        v21 = *v18 + v20;
        v22 = v20 >> 5;
        v23 = v20 >> 7;
        if ( !v23 )
          break;
        v24 = v19 + (v23 << 7);
        while ( *(_QWORD *)(v19 + 16) != a2 || *(_DWORD *)(v19 + 24) != *(_DWORD *)(v16 + 24) )
        {
          if ( *(_QWORD *)(v19 + 48) == a2 && *(_DWORD *)(v19 + 56) == *(_DWORD *)(v16 + 24) )
          {
            v19 += 32;
            break;
          }
          if ( *(_QWORD *)(v19 + 80) == a2 && *(_DWORD *)(v19 + 88) == *(_DWORD *)(v16 + 24) )
          {
            v19 += 64;
            break;
          }
          if ( *(_QWORD *)(v19 + 112) == a2 && *(_DWORD *)(v19 + 120) == *(_DWORD *)(v16 + 24) )
          {
            v19 += 96;
            break;
          }
          v19 += 128;
          if ( v19 == v24 )
          {
            v22 = (v21 - v19) >> 5;
            goto LABEL_26;
          }
        }
LABEL_52:
        if ( v21 == v19 )
          goto LABEL_29;
LABEL_53:
        v16 += 32LL;
        if ( (_BYTE *)v16 == v17 )
        {
LABEL_42:
          v29 = v44;
          v16 = (unsigned __int64)&v44[32 * (unsigned int)v45];
          if ( v44 != (_BYTE *)v16 )
          {
            do
            {
              v30 = *(_QWORD *)(v16 - 16);
              v16 -= 32LL;
              if ( v30 != 0 && v30 != -8 && v30 != -16 )
                sub_1649B30(v16);
            }
            while ( v29 != (_BYTE *)v16 );
            v16 = (unsigned __int64)v44;
          }
          goto LABEL_48;
        }
      }
LABEL_26:
      if ( v22 != 2 )
      {
        if ( v22 != 3 )
        {
          if ( v22 != 1 )
            goto LABEL_29;
          goto LABEL_78;
        }
        if ( *(_QWORD *)(v19 + 16) == a2 && *(_DWORD *)(v19 + 24) == *(_DWORD *)(v16 + 24) )
          goto LABEL_52;
        v19 += 32;
      }
      if ( *(_QWORD *)(v19 + 16) == a2 && *(_DWORD *)(v19 + 24) == *(_DWORD *)(v16 + 24) )
        goto LABEL_52;
      v19 += 32;
LABEL_78:
      if ( *(_QWORD *)(v19 + 16) == a2 && *(_DWORD *)(v19 + 24) == *(_DWORD *)(v16 + 24) )
        goto LABEL_52;
LABEL_29:
      v41[0] = (__int64 *)6;
      v41[1] = 0;
      v42 = a2;
      if ( a2 != -8 && a2 != -16 )
      {
        v37 = v18;
        sub_164C220(v41);
        v18 = v37;
      }
      v43 = *(_DWORD *)(v16 + 24);
      v25 = *((_DWORD *)v18 + 2);
      if ( v25 >= *((_DWORD *)v18 + 3) )
      {
        v39 = v18;
        sub_14CB640((__int64)v18, 0);
        v18 = v39;
        v25 = *((_DWORD *)v39 + 2);
      }
      v26 = *v18 + 32LL * v25;
      if ( v26 )
      {
        *(_QWORD *)v26 = 6;
        *(_QWORD *)(v26 + 8) = 0;
        v27 = v42;
        v28 = v42 == -8;
        *(_QWORD *)(v26 + 16) = v42;
        if ( v27 != 0 && !v28 && v27 != -16 )
        {
          v35 = v18;
          v38 = v26;
          sub_1649AC0(v26, (unsigned __int64)v41[0] & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v35;
          v26 = v38;
        }
        *(_DWORD *)(v26 + 24) = v43;
        v25 = *((_DWORD *)v18 + 2);
      }
      *((_DWORD *)v18 + 2) = v25 + 1;
      if ( v42 == 0 || v42 == -8 || v42 == -16 )
        goto LABEL_53;
      v16 += 32LL;
      sub_1649B30(v41);
      if ( (_BYTE *)v16 == v17 )
        goto LABEL_42;
    }
  }
LABEL_48:
  if ( (_BYTE *)v16 != v46 )
    _libc_free(v16);
}
