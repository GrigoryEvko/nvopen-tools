// Function: sub_F8F6B0
// Address: 0xf8f6b0
//
void __fastcall sub_F8F6B0(__int64 a1, _BYTE *a2)
{
  __int64 *v3; // r8
  __int64 v4; // rax
  __int64 *v5; // r12
  __int64 *v6; // r15
  __int64 v7; // rdx
  int v8; // eax
  unsigned __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rcx
  _BYTE *v14; // rdi
  _BYTE *v15; // [rsp-68h] [rbp-68h] BYREF
  __int64 v16; // [rsp-60h] [rbp-60h]
  _BYTE v17[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( !*(_BYTE *)(a1 + 64) )
  {
    v16 = 0x400000000LL;
    v3 = *(__int64 **)(a1 + 16);
    v4 = *(unsigned int *)(a1 + 24);
    v15 = v17;
    v5 = &v3[v4];
    if ( v3 == v5 )
    {
      v14 = v17;
    }
    else
    {
      v6 = v3;
      do
      {
        a2 = 0;
        v9 = sub_B46BC0(*v6, 0);
        if ( !v9 )
        {
          *(_BYTE *)(a1 + 64) = 1;
          v14 = v15;
          goto LABEL_10;
        }
        v12 = (unsigned int)v16;
        v13 = HIDWORD(v16);
        if ( (unsigned __int64)(unsigned int)v16 + 1 > HIDWORD(v16) )
        {
          a2 = v17;
          sub_C8D5F0((__int64)&v15, v17, (unsigned int)v16 + 1LL, 8u, v10, v11);
          v12 = (unsigned int)v16;
        }
        v7 = (__int64)v15;
        ++v6;
        *(_QWORD *)&v15[8 * v12] = v9;
        v8 = v16 + 1;
        LODWORD(v16) = v16 + 1;
      }
      while ( v5 != v6 );
      if ( v8 )
      {
        a2 = &v15;
        sub_F8EEF0(a1 + 16, (__int64)&v15, v7, v13, v10, v11);
        v14 = v15;
LABEL_10:
        if ( v14 != v17 )
          _libc_free(v14, a2);
        return;
      }
      v14 = v15;
    }
    *(_BYTE *)(a1 + 64) = 1;
    goto LABEL_10;
  }
}
