// Function: sub_9C66D0
// Address: 0x9c66d0
//
__int64 __fastcall sub_9C66D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r13d
  __int64 v6; // rbx
  unsigned int v7; // r14d
  unsigned __int64 v8; // rsi
  char v9; // dl
  __int64 v11; // r8
  _QWORD *v12; // rdi
  unsigned __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 *v15; // r9
  unsigned int v16; // r15d
  unsigned __int64 v17; // rsi
  int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // r14
  unsigned __int64 v23; // rax
  __int64 v24; // [rsp+0h] [rbp-E0h]
  int v25; // [rsp+18h] [rbp-C8h]
  __int64 v26; // [rsp+28h] [rbp-B8h]
  _QWORD *v27; // [rsp+30h] [rbp-B0h] BYREF
  const char *v28; // [rsp+38h] [rbp-A8h]
  _QWORD v29[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v30; // [rsp+50h] [rbp-90h] BYREF
  const char *v31; // [rsp+58h] [rbp-88h]
  _QWORD v32[2]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v33; // [rsp+70h] [rbp-70h] BYREF
  __int64 v34; // [rsp+78h] [rbp-68h]
  __int64 v35; // [rsp+80h] [rbp-60h]
  __int64 v36; // [rsp+88h] [rbp-58h]
  __int64 v37; // [rsp+90h] [rbp-50h]
  __int64 v38; // [rsp+98h] [rbp-48h]
  _QWORD *v39; // [rsp+A0h] [rbp-40h]

  v4 = a3;
  v6 = a2;
  v7 = *(_DWORD *)(a2 + 32);
  if ( v7 < (unsigned int)a3 )
  {
    v11 = 0;
    if ( v7 )
      v11 = *(_QWORD *)(a2 + 24);
    v12 = *(_QWORD **)(a2 + 16);
    v13 = *(_QWORD *)(a2 + 8);
    if ( (unsigned __int64)v12 >= v13 )
    {
      v24 = v11;
      v26 = ((__int64 (*)(void))sub_2241E50)();
      v27 = v29;
      v38 = 0x100000000LL;
      v39 = &v27;
      v33 = (unsigned __int64)&unk_49DD210;
      v28 = 0;
      LOBYTE(v29[0]) = 0;
      v34 = 0;
      v35 = 0;
      v36 = 0;
      v37 = 0;
      sub_CB5980(&v33, 0, 0, 0);
      v31 = "Unexpected end of file reading %u of %u bytes";
      v32[0] = v13;
      v30 = &unk_49D98C0;
      v32[1] = *(_QWORD *)(a2 + 16);
      sub_CB6620(&v33, &v30);
      v33 = (unsigned __int64)&unk_49DD210;
      sub_CB5840(&v33);
      a2 = (__int64)&v27;
      sub_9C3320((__int64 *)&v33, (__int64)&v27, 5u, v26);
      v12 = v27;
      v11 = v24;
      if ( v27 != v29 )
      {
        a2 = v29[0] + 1LL;
        j_j___libc_free_0(v27, v29[0] + 1LL);
        v11 = v24;
      }
      v21 = v33 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v33 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v21;
        return a1;
      }
      v16 = *(_DWORD *)(v6 + 32);
    }
    else
    {
      v14 = v12 + 1;
      v15 = (_QWORD *)((char *)v12 + *(_QWORD *)a2);
      if ( v13 < (unsigned __int64)(v12 + 1) )
      {
        *(_QWORD *)(a2 + 24) = 0;
        v18 = v13 - (_DWORD)v12;
        if ( v18 )
        {
          v19 = 0;
          a2 = 0;
          do
          {
            v20 = *((unsigned __int8 *)v15 + v19);
            a4 = (unsigned int)(8 * v19++);
            a3 = v20 << a4;
            a2 |= a3;
            *(_QWORD *)(v6 + 24) = a2;
          }
          while ( v19 != v18 );
          v16 = 8 * v18;
          v14 = (_QWORD *)((char *)v12 + v19);
        }
        else
        {
          v16 = 0;
          v14 = v12;
        }
      }
      else
      {
        a3 = *v15;
        v16 = 64;
        *(_QWORD *)(a2 + 24) = *v15;
      }
      *(_QWORD *)(v6 + 16) = v14;
      *(_DWORD *)(v6 + 32) = v16;
    }
    if ( v4 - v7 > v16 )
    {
      v25 = v4 - v7;
      v22 = sub_2241E50(v12, a2, a3, a4, v11);
      v39 = &v30;
      LOBYTE(v32[0]) = 0;
      v38 = 0x100000000LL;
      v30 = v32;
      v33 = (unsigned __int64)&unk_49DD210;
      v31 = 0;
      v34 = 0;
      v35 = 0;
      v36 = 0;
      v37 = 0;
      sub_CB5980(&v33, 0, 0, 0);
      v28 = "Unexpected end of file reading %u of %u bits";
      LODWORD(v29[0]) = v25;
      v27 = &unk_49D98E0;
      HIDWORD(v29[0]) = *(_DWORD *)(v6 + 32);
      sub_CB6620(&v33, &v27);
      v33 = (unsigned __int64)&unk_49DD210;
      sub_CB5840(&v33);
      sub_9C3320((__int64 *)&v33, (__int64)&v30, 5u, v22);
      if ( v30 != v32 )
        j_j___libc_free_0(v30, v32[0] + 1LL);
      v23 = v33;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v23 & 0xFFFFFFFFFFFFFFFELL;
    }
    else
    {
      v17 = *(_QWORD *)(v6 + 24);
      *(_DWORD *)(v6 + 32) = v7 - v4 + v16;
      *(_QWORD *)(v6 + 24) = v17 >> ((unsigned __int8)v4 - (unsigned __int8)v7);
      *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
      *(_QWORD *)a1 = ((v17 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v7 - (unsigned __int8)v4 + 64))) << v7) | v11;
    }
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 24);
    *(_DWORD *)(v6 + 32) = v7 - a3;
    *(_QWORD *)(v6 + 24) = v8 >> a3;
    v9 = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = v8 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4));
    *(_BYTE *)(a1 + 8) = v9;
  }
  return a1;
}
