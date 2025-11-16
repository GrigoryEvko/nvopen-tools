// Function: sub_1530A50
// Address: 0x1530a50
//
void __fastcall sub_1530A50(_DWORD **a1, __int64 *a2)
{
  _BYTE *v2; // rcx
  __int64 v3; // rax
  __int64 v4; // r13
  bool v5; // zf
  int v6; // edx
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  _DWORD *v13; // rdi
  __int64 v14; // [rsp+10h] [rbp-250h]
  _BYTE *v15; // [rsp+20h] [rbp-240h] BYREF
  __int64 v16; // [rsp+28h] [rbp-238h]
  _BYTE v17[560]; // [rsp+30h] [rbp-230h] BYREF

  v2 = v17;
  v3 = *a2;
  v4 = a2[2];
  v15 = v17;
  v5 = *(_BYTE *)(v3 + 16) == 18;
  v16 = 0x4000000000LL;
  v6 = 0;
  v7 = v5 + 1;
  v8 = a2[3] - v4;
  v9 = v8 >> 2;
  if ( (unsigned __int64)v8 > 0x100 )
  {
    v14 = a2[3] - v4;
    sub_16CD150(&v15, v17, v9, 8);
    v6 = v16;
    v8 = v14;
    v2 = &v15[8 * (unsigned int)v16];
  }
  if ( v8 > 0 )
  {
    v10 = 0;
    do
    {
      *(_QWORD *)&v2[8 * v10] = *(unsigned int *)(v4 + 4 * v10);
      ++v10;
    }
    while ( v9 - v10 > 0 );
    v6 = v16;
  }
  LODWORD(v16) = v6 + v9;
  v11 = (unsigned int)sub_153E840(a1 + 3);
  v12 = (unsigned int)v16;
  if ( (unsigned int)v16 >= HIDWORD(v16) )
  {
    sub_16CD150(&v15, v17, 0, 8);
    v12 = (unsigned int)v16;
  }
  *(_QWORD *)&v15[8 * v12] = v11;
  v13 = *a1;
  LODWORD(v16) = v16 + 1;
  sub_152F3D0(v13, v7, (__int64)&v15, 0);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
}
