// Function: sub_30FBC60
// Address: 0x30fbc60
//
void __fastcall sub_30FBC60(__int64 a1, __int64 a2)
{
  char *v2; // rax
  __int64 v3; // rdx
  __int64 i; // rbx
  bool v5; // zf
  const char *v6; // rbx
  size_t v7; // rax
  unsigned int v8; // edx
  __int64 v9; // rcx
  _BYTE *v10[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v11; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v12; // [rsp+50h] [rbp-B0h]
  __int64 v13; // [rsp+58h] [rbp-A8h]
  __int64 v14; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v15; // [rsp+70h] [rbp-90h] BYREF
  __int64 v16[2]; // [rsp+80h] [rbp-80h] BYREF
  _BYTE v17[12]; // [rsp+90h] [rbp-70h] BYREF
  char v18; // [rsp+9Ch] [rbp-64h]
  _QWORD *v19; // [rsp+A0h] [rbp-60h] BYREF
  size_t v20; // [rsp+A8h] [rbp-58h]
  _QWORD v21[2]; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v22; // [rsp+C0h] [rbp-40h]

  v2 = (char *)sub_BD5D20(*(_QWORD *)(a1 + 24));
  sub_B16430((__int64)v16, "Callee", 6u, v2, v3);
  sub_30FB310(a2, (__int64)v16);
  if ( v19 != v21 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( (_BYTE *)v16[0] != v17 )
    j_j___libc_free_0(v16[0]);
  for ( i = 0; i != 304; i += 8 )
  {
    sub_B167F0(
      (__int64 *)v10,
      *(_BYTE **)(unk_5031590 + 10 * i),
      *(_QWORD *)(unk_5031590 + 10 * i + 8),
      **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL) + 24LL) + i));
    v16[0] = (__int64)v17;
    sub_30FA730(v16, v10[0], (__int64)&v10[0][(unsigned __int64)v10[1]]);
    v19 = v21;
    sub_30FA730((__int64 *)&v19, v12, (__int64)v12 + v13);
    v22 = _mm_loadu_si128(&v15);
    sub_B180C0(a2, (unsigned __int64)v16);
    if ( v19 != v21 )
      j_j___libc_free_0((unsigned __int64)v19);
    if ( (_BYTE *)v16[0] != v17 )
      j_j___libc_free_0(v16[0]);
    if ( v12 != &v14 )
      j_j___libc_free_0((unsigned __int64)v12);
    if ( (__int64 *)v10[0] != &v11 )
      j_j___libc_free_0((unsigned __int64)v10[0]);
  }
  v5 = *(_BYTE *)(a1 + 56) == 0;
  v6 = "true";
  v16[1] = 12;
  v16[0] = (__int64)v17;
  qmemcpy(v17, "ShouldInline", sizeof(v17));
  if ( v5 )
    v6 = "false";
  v18 = 0;
  v19 = v21;
  v7 = strlen(v6);
  if ( (unsigned int)v7 < 8 )
  {
    if ( (v7 & 4) != 0 )
    {
      LODWORD(v21[0]) = *(_DWORD *)v6;
      *(_DWORD *)((char *)&v20 + (unsigned int)v7 + 4) = *(_DWORD *)&v6[(unsigned int)v7 - 4];
    }
    else if ( (_DWORD)v7 )
    {
      LOBYTE(v21[0]) = *v6;
      if ( (v7 & 2) != 0 )
        *(_WORD *)((char *)&v20 + (unsigned int)v7 + 6) = *(_WORD *)&v6[(unsigned int)v7 - 2];
    }
  }
  else
  {
    *(_QWORD *)((char *)&v21[-1] + (unsigned int)v7) = *(_QWORD *)&v6[(unsigned int)v7 - 8];
    if ( (unsigned int)(v7 - 1) >= 8 )
    {
      v8 = 0;
      do
      {
        v9 = v8;
        v8 += 8;
        *(_QWORD *)((char *)v21 + v9) = *(_QWORD *)&v6[v9];
      }
      while ( v8 < (((_DWORD)v7 - 1) & 0xFFFFFFF8) );
    }
  }
  v20 = v7;
  *((_BYTE *)v21 + v7) = 0;
  v22 = 0u;
  sub_30FB310(a2, (__int64)v16);
  if ( v19 != v21 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( (_BYTE *)v16[0] != v17 )
    j_j___libc_free_0(v16[0]);
}
