// Function: sub_B0E210
// Address: 0xb0e210
//
__int64 __fastcall sub_B0E210(__int64 a1, const void *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  char v7; // r8
  signed __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // rdx
  const void *v11; // rsi
  __int64 v12; // r12
  unsigned __int64 v14; // rdx
  char v15; // [rsp+Fh] [rbp-E1h]
  char v16; // [rsp+Fh] [rbp-E1h]
  _BYTE v17[16]; // [rsp+10h] [rbp-E0h] BYREF
  char v18; // [rsp+20h] [rbp-D0h]
  _QWORD *v19; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+38h] [rbp-B8h]
  _QWORD v21[22]; // [rsp+40h] [rbp-B0h] BYREF

  sub_AF47B0((__int64)v17, *(unsigned __int64 **)(a1 + 16), *(unsigned __int64 **)(a1 + 24));
  v5 = *(_QWORD *)(a1 + 16);
  v6 = ((*(_QWORD *)(a1 + 24) - v5) >> 3) - (v18 != 0 ? 3 : 0);
  if ( (v18 != 0 ? 3 : 0) >= (unsigned int)((*(_QWORD *)(a1 + 24) - v5) >> 3) || *(_QWORD *)(v5 + 8 * v6 - 8) == 159 )
  {
    v19 = v21;
    if ( v6 )
    {
      v7 = 0;
      v20 = 0x1000000000LL;
      v6 = 0;
    }
    else
    {
      v7 = 1;
      v20 = 0x1000000000LL;
    }
  }
  else
  {
    v19 = v21;
    v7 = 1;
    v20 = 0x1000000001LL;
    v6 = 1;
    v21[0] = 6;
  }
  v8 = 8 * a3;
  v9 = (v8 >> 3) + v6;
  if ( v9 > 0x10 )
  {
    v16 = v7;
    sub_C8D5F0(&v19, v21, v9, 8);
    v6 = (unsigned int)v20;
    v7 = v16;
  }
  if ( v8 )
  {
    v15 = v7;
    memcpy(&v19[v6], a2, v8);
    LODWORD(v6) = v20;
    v7 = v15;
  }
  LODWORD(v6) = (v8 >> 3) + v6;
  LODWORD(v20) = v6;
  v10 = (unsigned int)v6;
  if ( v7 )
  {
    v6 = (unsigned int)v6;
    v14 = (unsigned int)v6 + 1LL;
    if ( v14 > HIDWORD(v20) )
    {
      sub_C8D5F0(&v19, v21, v14, 8);
      v6 = (unsigned int)v20;
    }
    v19[v6] = 159;
    v10 = (unsigned int)(v20 + 1);
    LODWORD(v20) = v20 + 1;
  }
  v11 = v19;
  v12 = sub_B0DED0((_QWORD *)a1, v19, v10);
  if ( v19 != v21 )
    _libc_free(v19, v11);
  return v12;
}
