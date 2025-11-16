// Function: sub_C5F090
// Address: 0xc5f090
//
__int64 __fastcall sub_C5F090(__int64 *a1, unsigned __int64 a2, unsigned __int64 *a3)
{
  _QWORD *v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  __int64 v10; // r10
  _BYTE *v11; // rdi
  __int64 v12; // rcx
  _BYTE *v13; // r9
  _BYTE *v14; // rax
  const char *v15; // r15
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  _QWORD v18[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v20[4]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v21[14]; // [rsp+60h] [rbp-70h] BYREF

  v4 = (_QWORD *)a2;
  v5 = *a1;
  v6 = a1[1];
  if ( a3 )
  {
    v7 = *a3 & 0xFFFFFFFFFFFFFFFELL;
    if ( v7 )
    {
      v8 = 0;
LABEL_4:
      *a3 = v7 | 1;
      return v8;
    }
    *a3 = 0;
  }
  v10 = *(_QWORD *)a2;
  v11 = (_BYTE *)(v5 + v6);
  v12 = 0;
  v8 = 0;
  v13 = (_BYTE *)(v5 + *(_QWORD *)a2);
  v14 = v13;
  do
  {
    if ( v11 == v14 )
    {
      v15 = "malformed sleb128, extends past end";
LABEL_12:
      v8 = 0;
      if ( !a3 )
        return v8;
      v16 = sub_2241E50(v11, a2, v5, v12, 0);
      v18[0] = v19;
      v21[5] = 0x100000000LL;
      v21[6] = (__int64)v18;
      v21[0] = (__int64)&unk_49DD210;
      v18[1] = 0;
      LOBYTE(v19[0]) = 0;
      memset(&v21[1], 0, 32);
      sub_CB5980(v21, 0, 0, 0);
      v20[2] = v15;
      v20[1] = "unable to decode LEB128 at offset 0x%8.8lx: %s";
      v20[0] = &unk_49DC5E0;
      v20[3] = *v4;
      sub_CB6620(v21, v20);
      v21[0] = (__int64)&unk_49DD210;
      sub_CB5840(v21);
      sub_C5E9B0(v21, (__int64)v18, 0x54u, v16);
      if ( (_QWORD *)v18[0] != v19 )
        j_j___libc_free_0(v18[0], v19[0] + 1LL);
      v17 = *a3;
      if ( (*a3 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(a3);
      v8 = 0;
      v7 = (v21[0] | v17) & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_18;
    }
    a2 = (unsigned __int8)*v14;
    v5 = *v14 & 0x7F;
    if ( (unsigned int)v12 > 0x3E )
    {
      if ( v5 != 0 && v5 != 127 )
      {
        if ( (_DWORD)v12 == 63 )
          goto LABEL_23;
      }
      else if ( (_DWORD)v12 == 63 )
      {
        goto LABEL_9;
      }
      if ( v5 != ((v8 >> 63) & 0x7F) )
      {
LABEL_23:
        v15 = "sleb128 too big for int64";
        goto LABEL_12;
      }
    }
LABEL_9:
    v5 <<= v12;
    ++v14;
    v12 = (unsigned int)(v12 + 7);
    v8 |= v5;
  }
  while ( (a2 & 0x80u) != 0LL );
  if ( (unsigned int)v12 <= 0x3F && (a2 & 0x40) != 0 )
    v8 |= -1LL << v12;
  *v4 = v10 + (unsigned int)((_DWORD)v14 - (_DWORD)v13);
  if ( !a3 )
    return v8;
  v7 = *a3 & 0xFFFFFFFFFFFFFFFELL;
LABEL_18:
  if ( v7 )
    goto LABEL_4;
  *a3 = 1;
  return v8;
}
