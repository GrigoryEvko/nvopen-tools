// Function: sub_15C4CE0
// Address: 0x15c4ce0
//
__int64 __fastcall sub_15C4CE0(__int64 a1, const void *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  signed __int64 v8; // r14
  char v9; // r15
  __int64 v10; // r8
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r12
  int v15; // [rsp+8h] [rbp-E8h]
  int v16; // [rsp+8h] [rbp-E8h]
  _BYTE v17[16]; // [rsp+10h] [rbp-E0h] BYREF
  char v18; // [rsp+20h] [rbp-D0h]
  _QWORD *v19; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+38h] [rbp-B8h]
  _QWORD v21[22]; // [rsp+40h] [rbp-B0h] BYREF

  sub_15B1350((__int64)v17, *(unsigned __int64 **)(a1 + 24), *(unsigned __int64 **)(a1 + 32));
  v5 = *(_QWORD *)(a1 + 24);
  v6 = ((*(_QWORD *)(a1 + 32) - v5) >> 3) - (v18 != 0 ? 3 : 0);
  if ( (v18 != 0 ? 3 : 0) >= (unsigned int)((*(_QWORD *)(a1 + 32) - v5) >> 3) || *(_QWORD *)(v5 + 8 * v6 - 8) == 159 )
  {
    v19 = v21;
    v20 = 0x1000000000LL;
    v7 = 16;
    if ( v6 )
    {
      v8 = 8 * a3;
      v6 = 0;
      v9 = 0;
      v10 = v8 >> 3;
      if ( (unsigned __int64)(v8 >> 3) <= 0x10 )
        goto LABEL_5;
      goto LABEL_13;
    }
  }
  else
  {
    v19 = v21;
    v6 = 1;
    v21[0] = 6;
    v20 = 0x1000000001LL;
    v7 = 15;
  }
  v8 = 8 * a3;
  v9 = 1;
  v10 = v8 >> 3;
  if ( v8 >> 3 <= v7 )
    goto LABEL_5;
LABEL_13:
  v16 = v10;
  sub_16CD150(&v19, v21, v10 + v6, 8);
  v6 = (unsigned int)v20;
  LODWORD(v10) = v16;
LABEL_5:
  if ( v8 )
  {
    v15 = v10;
    memcpy(&v19[v6], a2, v8);
    LODWORD(v6) = v20;
    LODWORD(v10) = v15;
  }
  v11 = v10 + v6;
  LODWORD(v20) = v11;
  v12 = v11;
  if ( v9 )
  {
    if ( v11 >= HIDWORD(v20) )
    {
      sub_16CD150(&v19, v21, 0, 8);
      v12 = (unsigned int)v20;
    }
    v19[v12] = 159;
    LODWORD(v12) = v20 + 1;
    LODWORD(v20) = v20 + 1;
  }
  v13 = sub_15C49B0((_QWORD *)a1, v19, (unsigned int)v12);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  return v13;
}
