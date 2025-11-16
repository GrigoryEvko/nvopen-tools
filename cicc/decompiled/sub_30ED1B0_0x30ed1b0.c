// Function: sub_30ED1B0
// Address: 0x30ed1b0
//
void __fastcall sub_30ED1B0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int8 *a4, size_t a5)
{
  bool v8; // zf
  __int64 v9; // rax
  size_t v10; // rdi
  unsigned __int8 v11; // dl
  __int64 v12; // rax
  __int64 v13; // rdi
  const void *v14; // rax
  size_t v15; // rdx
  __int64 v16; // r9
  size_t v17; // rdi
  size_t v18; // r8
  const void *v19; // [rsp+0h] [rbp-110h]
  size_t v20; // [rsp+8h] [rbp-108h]
  size_t na; // [rsp+10h] [rbp-100h]
  size_t nb; // [rsp+10h] [rbp-100h]
  size_t n; // [rsp+10h] [rbp-100h]
  _QWORD v25[8]; // [rsp+20h] [rbp-F0h] BYREF
  __int8 *v26; // [rsp+60h] [rbp-B0h] BYREF
  size_t v27; // [rsp+68h] [rbp-A8h]
  unsigned __int64 v28; // [rsp+70h] [rbp-A0h]
  _BYTE v29[152]; // [rsp+78h] [rbp-98h] BYREF

  v8 = *a3 == 0;
  v26 = v29;
  v27 = 0;
  v28 = 100;
  if ( !v8 )
    goto LABEL_18;
  v9 = sub_B92180((__int64)a3);
  v10 = v27;
  if ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 32) & 0x40) != 0 )
    {
      n = v9;
      sub_B18290(a1, "artificial ", 0xBu);
      v9 = n;
    }
    v11 = *(_BYTE *)(v9 - 16);
    if ( (v11 & 2) != 0 )
      v12 = *(_QWORD *)(v9 - 32);
    else
      v12 = v9 - 16 - 8LL * ((v11 >> 2) & 0xF);
    v13 = *(_QWORD *)(v12 + 16);
    if ( v13 )
    {
      v14 = (const void *)sub_B91420(v13);
      v17 = 0;
      v27 = 0;
      v18 = v15;
      if ( v28 < v15 )
      {
        v19 = v14;
        v20 = v15;
        na = v15;
        sub_C8D290((__int64)&v26, v29, v15, 1u, v15, v16);
        v17 = v27;
        v14 = v19;
        v18 = v20;
        v15 = na;
      }
      if ( v15 )
      {
        nb = v18;
        memcpy(&v26[v17], v14, v15);
        v17 = v27;
        v18 = nb;
      }
    }
    else
    {
      v18 = 0;
      v17 = 0;
    }
    v10 = v18 + v17;
    v27 = v10;
  }
  if ( v10 )
  {
    if ( !a5 )
      goto LABEL_15;
  }
  else
  {
LABEL_18:
    v25[5] = 0x100000000LL;
    v25[1] = 2;
    v25[0] = &unk_49DD288;
    v25[6] = &v26;
    memset(&v25[2], 0, 24);
    sub_CB5980((__int64)v25, 0, 0, 0);
    sub_A5BF40(a3, (__int64)v25, 0, a2);
    v25[0] = &unk_49DD388;
    sub_CB5840((__int64)v25);
    if ( !a5 )
      goto LABEL_15;
  }
  sub_B18290(a1, a4, a5);
  sub_B18290(a1, " ", 1u);
LABEL_15:
  sub_B18290(a1, "'", 1u);
  sub_B18290(a1, v26, v27);
  sub_B18290(a1, "'", 1u);
  if ( v26 != v29 )
    _libc_free((unsigned __int64)v26);
}
