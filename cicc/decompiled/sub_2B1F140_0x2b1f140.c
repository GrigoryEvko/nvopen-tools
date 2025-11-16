// Function: sub_2B1F140
// Address: 0x2b1f140
//
__int64 __fastcall sub_2B1F140(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  unsigned int v5; // r12d
  unsigned int v6; // ebx
  __int64 v7; // rdi
  int v8; // edx
  int v9; // eax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  size_t v15; // r8
  __int64 v16; // r9
  _BYTE *v17; // rcx
  _DWORD *v18; // rax
  __int64 **v19; // rdi
  __int64 v20; // rbx
  _QWORD *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  void *v26; // [rsp+0h] [rbp-F0h]
  _BYTE *v27; // [rsp+0h] [rbp-F0h]
  size_t n; // [rsp+8h] [rbp-E8h]
  __int64 na; // [rsp+8h] [rbp-E8h]
  __int64 v30; // [rsp+18h] [rbp-D8h]
  _QWORD v31[4]; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v32; // [rsp+40h] [rbp-B0h]
  _QWORD v33[4]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v34; // [rsp+70h] [rbp-80h]
  void *v35; // [rsp+80h] [rbp-70h] BYREF
  __int64 v36; // [rsp+88h] [rbp-68h]
  _BYTE s[16]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v38; // [rsp+A0h] [rbp-50h]

  v5 = a3;
  v6 = a4;
  if ( !(a4 % a3) )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(unsigned __int8 *)(v7 + 8);
    if ( (unsigned int)(v8 - 17) > 1 )
    {
      if ( (_BYTE)v8 != 17 )
        goto LABEL_6;
    }
    else
    {
      v7 = **(_QWORD **)(v7 + 16);
      v9 = *(unsigned __int8 *)(v7 + 8);
      if ( (_BYTE)v9 != 17 )
      {
        if ( (unsigned int)(v9 - 17) > 1 )
          goto LABEL_6;
        goto LABEL_5;
      }
    }
    v5 *= *(_DWORD *)(v7 + 32);
LABEL_5:
    v7 = **(_QWORD **)(v7 + 16);
LABEL_6:
    v38 = 257;
    v10 = sub_BCDA70((__int64 *)v7, v5);
    v11 = sub_BCB2E0(*(_QWORD **)(a1 + 72));
    BYTE4(v30) = 0;
    v33[1] = sub_ACD640(v11, v6, 0);
    v12 = *(_QWORD *)(a2 + 8);
    v31[0] = v10;
    v33[0] = a2;
    v31[1] = v12;
    return sub_B33D10(a1, 0x17Du, (__int64)v31, 2, (int)v33, 2, v30, (__int64)&v35);
  }
  v15 = a3;
  v35 = s;
  v16 = 4LL * a3;
  v36 = 0xC00000000LL;
  if ( a3 > 0xCuLL )
  {
    na = 4LL * a3;
    sub_C8D5F0((__int64)&v35, s, a3, 4u, a3, v16);
    memset(v35, 255, na);
    v18 = v35;
    LODWORD(v36) = v5;
    v15 = v5;
    v17 = (char *)v35 + na;
  }
  else
  {
    v17 = &s[v16];
    if ( a3 && v17 != s )
    {
      v27 = &s[v16];
      memset(s, 255, 4LL * a3);
      v17 = v27;
      v15 = v5;
    }
    LODWORD(v36) = v5;
    v18 = s;
  }
  if ( v18 != (_DWORD *)v17 )
  {
    do
      *v18++ = v6++;
    while ( v18 != (_DWORD *)v17 );
    v17 = v35;
    v15 = (unsigned int)v36;
  }
  v19 = *(__int64 ***)(a2 + 8);
  v26 = v17;
  n = v15;
  v32 = 257;
  v20 = sub_ACADE0(v19);
  v13 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, void *, size_t))(**(_QWORD **)(a1 + 80) + 112LL))(
          *(_QWORD *)(a1 + 80),
          a2,
          v20,
          v26,
          n);
  if ( !v13 )
  {
    v34 = 257;
    v21 = sub_BD2C40(112, unk_3F1FE60);
    v13 = (__int64)v21;
    if ( v21 )
      sub_B4E9E0((__int64)v21, a2, v20, v26, n, (__int64)v33, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v13,
      v31,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v22 = *(_QWORD *)a1;
    v23 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    while ( v23 != v22 )
    {
      v24 = *(_QWORD *)(v22 + 8);
      v25 = *(_DWORD *)v22;
      v22 += 16;
      sub_B99FD0(v13, v25, v24);
    }
  }
  if ( v35 != s )
    _libc_free((unsigned __int64)v35);
  return v13;
}
