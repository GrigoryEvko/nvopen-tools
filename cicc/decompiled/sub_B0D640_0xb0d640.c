// Function: sub_B0D640
// Address: 0xb0d640
//
__int64 __fastcall sub_B0D640(_QWORD *a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 *v3; // r15
  unsigned __int64 *v4; // r12
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int64 *v11; // r15
  int v12; // eax
  unsigned __int64 *v13; // r9
  unsigned __int64 *v14; // r10
  __int64 v15; // rax
  size_t v16; // r11
  __int64 v17; // r15
  unsigned __int64 v18; // rdx
  __int64 *v19; // rsi
  __int64 v20; // rdx
  __int64 *v21; // rdi
  __int64 v22; // r12
  size_t v25; // [rsp+8h] [rbp-B8h]
  unsigned __int64 *v26; // [rsp+10h] [rbp-B0h]
  unsigned __int64 *v27; // [rsp+18h] [rbp-A8h]
  void *src; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 *v30; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v31; // [rsp+40h] [rbp-80h] BYREF
  __int64 v32; // [rsp+48h] [rbp-78h]
  _BYTE v33[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = (unsigned __int64 *)a1[2];
  v4 = (unsigned __int64 *)a1[3];
  v31 = (__int64 *)v33;
  v32 = 0x800000000LL;
  v30 = v3;
  if ( v3 == v4 )
  {
    v19 = (__int64 *)v33;
    v20 = 0;
  }
  else
  {
    do
    {
      src = v30;
      if ( *v3 == 4101 && a2 <= v3[1] )
      {
        v5 = (unsigned int)v32;
        v6 = (unsigned int)v32 + 1LL;
        if ( v6 > HIDWORD(v32) )
        {
          sub_C8D5F0(&v31, v33, v6, 8);
          v5 = (unsigned int)v32;
        }
        v31[v5] = 4101;
        LODWORD(v32) = v32 + 1;
        v7 = (unsigned int)v32;
        v8 = *((_QWORD *)src + 1);
        v9 = (unsigned int)v32 + 1LL;
        if ( a2 == v8 )
          v8 = a3;
        v10 = (__PAIR128__(v8, a2) - v8) >> 64;
        if ( v9 > HIDWORD(v32) )
        {
          sub_C8D5F0(&v31, v33, v9, 8);
          v7 = (unsigned int)v32;
        }
        v31[v7] = v10;
        LODWORD(v32) = v32 + 1;
      }
      else
      {
        v12 = sub_AF4160((unsigned __int64 **)&src);
        v13 = (unsigned __int64 *)src;
        v14 = &v3[v12];
        v15 = (unsigned int)v32;
        v16 = (char *)v14 - (_BYTE *)src;
        v17 = ((char *)v14 - (_BYTE *)src) >> 3;
        v18 = v17 + (unsigned int)v32;
        if ( v18 > HIDWORD(v32) )
        {
          v25 = (char *)v14 - (_BYTE *)src;
          v26 = (unsigned __int64 *)src;
          v27 = v14;
          sub_C8D5F0(&v31, v33, v18, 8);
          v15 = (unsigned int)v32;
          v16 = v25;
          v13 = v26;
          v14 = v27;
        }
        if ( v14 != v13 )
        {
          memcpy(&v31[v15], v13, v16);
          LODWORD(v15) = v32;
        }
        LODWORD(v32) = v17 + v15;
      }
      v11 = v30;
      v3 = &v11[(unsigned int)sub_AF4160(&v30)];
      v30 = v3;
    }
    while ( v4 != v3 );
    v19 = v31;
    v20 = (unsigned int)v32;
  }
  v21 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[1] & 4) != 0 )
    v21 = (__int64 *)*v21;
  v22 = sub_B0D000(v21, v19, v20, 0, 1);
  if ( v31 != (__int64 *)v33 )
    _libc_free(v31, v19);
  return v22;
}
