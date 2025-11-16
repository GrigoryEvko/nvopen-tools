// Function: sub_A20640
// Address: 0xa20640
//
void __fastcall sub_A20640(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // rcx
  __int64 v3; // r13
  bool v4; // zf
  int v5; // edx
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // [rsp+10h] [rbp-250h]
  _BYTE *v15; // [rsp+20h] [rbp-240h] BYREF
  __int64 v16; // [rsp+28h] [rbp-238h]
  _BYTE v17[560]; // [rsp+30h] [rbp-230h] BYREF

  v2 = v17;
  v3 = *(_QWORD *)(a2 + 16);
  v4 = **(_BYTE **)a2 == 23;
  v15 = v17;
  v16 = 0x4000000000LL;
  v5 = 0;
  v6 = v4 + 1;
  v7 = *(_QWORD *)(a2 + 24) - v3;
  v8 = v7 >> 2;
  if ( (unsigned __int64)v7 > 0x100 )
  {
    v14 = *(_QWORD *)(a2 + 24) - v3;
    sub_C8D5F0(&v15, v17, v8, 8);
    v5 = v16;
    v7 = v14;
    v2 = &v15[8 * (unsigned int)v16];
  }
  if ( v7 > 0 )
  {
    v9 = 0;
    do
    {
      *(_QWORD *)&v2[8 * v9] = *(unsigned int *)(v3 + 4 * v9);
      ++v9;
    }
    while ( v8 - v9 > 0 );
    v5 = v16;
  }
  LODWORD(v16) = v5 + v8;
  v10 = sub_A3F3B0(a1 + 3);
  v11 = (unsigned int)v16;
  v12 = v10;
  if ( (unsigned __int64)(unsigned int)v16 + 1 > HIDWORD(v16) )
  {
    sub_C8D5F0(&v15, v17, (unsigned int)v16 + 1LL, 8);
    v11 = (unsigned int)v16;
  }
  *(_QWORD *)&v15[8 * v11] = v12;
  v13 = *a1;
  LODWORD(v16) = v16 + 1;
  sub_A1FB70(v13, v6, (__int64)&v15, 0);
  if ( v15 != v17 )
    _libc_free(v15, v6);
}
