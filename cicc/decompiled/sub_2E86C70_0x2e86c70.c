// Function: sub_2E86C70
// Address: 0x2e86c70
//
void __fastcall sub_2E86C70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, void *a6)
{
  __int64 v8; // rdx
  int *v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // r10
  size_t v14; // r15
  _BYTE *v15; // rdi
  __int64 v16; // rdx
  void *src; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  _QWORD *v21; // [rsp+20h] [rbp-50h] BYREF
  __int64 v22; // [rsp+28h] [rbp-48h]
  _BYTE v23[64]; // [rsp+30h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a1 + 48);
  v22 = 0x200000000LL;
  v21 = v23;
  v9 = (int *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v10 = 0;
    goto LABEL_3;
  }
  v12 = v8 & 7;
  if ( v12 )
  {
    if ( v12 != 3 )
    {
      v13 = 0;
      v10 = 0;
      v14 = 0;
      a6 = 0;
      goto LABEL_17;
    }
    v16 = *v9;
    a6 = v9 + 4;
    v13 = (__int64)&v9[2 * v16 + 4];
    v14 = 8 * v16;
    v10 = (8 * v16) >> 3;
  }
  else
  {
    v13 = a1 + 56;
    a6 = (void *)(a1 + 48);
    *(_QWORD *)(a1 + 48) = v9;
    v14 = 8;
    v10 = 1;
  }
  if ( v14 > 0x10 )
  {
    src = a6;
    v18 = v13;
    v19 = v10;
    sub_C8D5F0((__int64)&v21, v23, v10, 8u, v10, (__int64)a6);
    LODWORD(v9) = v22;
    v10 = v19;
    v13 = v18;
    a6 = src;
    v15 = &v21[(unsigned int)v22];
    goto LABEL_12;
  }
LABEL_17:
  v15 = v23;
  LODWORD(v9) = 0;
LABEL_12:
  if ( (void *)v13 != a6 )
  {
    v20 = v10;
    memcpy(v15, a6, v14);
    LODWORD(v9) = v22;
    v10 = v20;
  }
LABEL_3:
  LODWORD(v22) = v10 + (_DWORD)v9;
  v11 = (unsigned int)(v10 + (_DWORD)v9);
  if ( v11 + 1 > (unsigned __int64)HIDWORD(v22) )
  {
    sub_C8D5F0((__int64)&v21, v23, v11 + 1, 8u, v10, (__int64)a6);
    v11 = (unsigned int)v22;
  }
  v21[v11] = a3;
  LODWORD(v22) = v22 + 1;
  sub_2E86A90(a1, a2, v21, (unsigned int)v22);
  if ( v21 != (_QWORD *)v23 )
    _libc_free((unsigned __int64)v21);
}
