// Function: sub_DAF6F0
// Address: 0xdaf6f0
//
__int64 __fastcall sub_DAF6F0(__int64 *a1, unsigned __int64 *a2, __int64 a3, __int16 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // r14
  _DWORD *v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 *v11; // rbx
  int v12; // eax
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  _DWORD **v15; // rsi
  __int64 v16; // r14
  _DWORD *v17; // rdi
  __int64 v19; // rdx
  unsigned __int64 *v20; // r14
  unsigned __int64 v21; // rcx
  char *v22; // r13
  __int64 v23; // rdx
  char *v24; // rbx
  __int64 v25; // rdx
  __int16 v26; // r15
  char *v27; // [rsp+10h] [rbp-140h]
  __int64 v28; // [rsp+18h] [rbp-138h]
  __int64 n; // [rsp+28h] [rbp-128h]
  void *na; // [rsp+28h] [rbp-128h]
  int desta; // [rsp+30h] [rbp-120h]
  char *dest; // [rsp+30h] [rbp-120h]
  __int64 *v35; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v36; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v37; // [rsp+68h] [rbp-E8h]
  __int64 v38; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v39; // [rsp+78h] [rbp-D8h]
  __int64 v40; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v41; // [rsp+88h] [rbp-C8h]
  _DWORD *v42; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+98h] [rbp-B8h]
  _DWORD v44[44]; // [rsp+A0h] [rbp-B0h] BYREF

  v6 = a2;
  v43 = 0x2000000001LL;
  v7 = &a2[a3];
  v42 = v44;
  v44[0] = 6;
  n = 8 * a3;
  if ( a2 != v7 )
  {
    v8 = *a2;
    v9 = v44;
    v10 = 1;
    v11 = a2 + 1;
    v12 = *a2;
    while ( 1 )
    {
      v9[v10] = v12;
      v13 = HIDWORD(v8);
      LODWORD(v43) = v43 + 1;
      v14 = (unsigned int)v43;
      if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
      {
        sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 4u, a5, a6);
        v14 = (unsigned int)v43;
      }
      v42[v14] = v13;
      v10 = (unsigned int)(v43 + 1);
      LODWORD(v43) = v43 + 1;
      if ( v7 == v11 )
        break;
      v8 = *v11;
      a5 = v10 + 1;
      v12 = *v11;
      if ( v10 + 1 > (unsigned __int64)HIDWORD(v43) )
      {
        desta = *v11;
        sub_C8D5F0((__int64)&v42, v44, v10 + 1, 4u, a5, a6);
        v10 = (unsigned int)v43;
        v12 = desta;
      }
      v9 = v42;
      ++v11;
    }
    v6 = a2;
  }
  v15 = &v42;
  v35 = 0;
  v16 = (__int64)sub_C65B40((__int64)(a1 + 129), (__int64)&v42, (__int64 *)&v35, (__int64)off_49DEA80);
  if ( !v16 )
  {
    v19 = a1[133];
    a1[143] += n;
    v20 = (unsigned __int64 *)(a1 + 133);
    v21 = n + ((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( a1[134] >= v21 && v19 )
    {
      a1[133] = v21;
      dest = (char *)((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      dest = (char *)sub_9D1E70((__int64)v20, n, n, 3);
    }
    if ( v6 != v7 )
      memmove(dest, v6, n);
    v22 = &dest[n];
    v27 = &dest[n];
    na = sub_C65D30((__int64)&v42, v20);
    v28 = v23;
    v16 = sub_A777F0(0x30u, (__int64 *)v20);
    if ( v16 )
    {
      v37 = 16;
      v36 = 1;
      if ( v22 == dest )
        goto LABEL_34;
      v24 = dest;
      do
      {
        v25 = *(_QWORD *)v24;
        v41 = 16;
        v40 = *(unsigned __int16 *)(v25 + 26);
        sub_C49B30((__int64)&v38, (__int64)&v36, &v40);
        if ( v37 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
        v36 = v38;
        v37 = v39;
        if ( v41 > 0x40 && v40 )
          j_j___libc_free_0_0(v40);
        v24 += 8;
      }
      while ( v27 != v24 );
      if ( v37 <= 0x40 )
      {
LABEL_34:
        v26 = v36;
      }
      else
      {
        v26 = *(_WORD *)v36;
        j_j___libc_free_0_0(v36);
      }
      *(_WORD *)(v16 + 26) = v26;
      *(_QWORD *)v16 = 0;
      *(_QWORD *)(v16 + 8) = na;
      *(_WORD *)(v16 + 28) = 0;
      *(_QWORD *)(v16 + 16) = v28;
      *(_WORD *)(v16 + 24) = 6;
      *(_QWORD *)(v16 + 32) = dest;
      *(_QWORD *)(v16 + 40) = a3;
    }
    sub_C657C0(a1 + 129, (__int64 *)v16, v35, (__int64)off_49DEA80);
    v15 = (_DWORD **)v16;
    sub_DAEE00((__int64)a1, v16, (__int64 *)v6, a3);
  }
  v17 = v42;
  *(_WORD *)(v16 + 28) |= a4;
  if ( v17 != v44 )
    _libc_free(v17, v15);
  return v16;
}
