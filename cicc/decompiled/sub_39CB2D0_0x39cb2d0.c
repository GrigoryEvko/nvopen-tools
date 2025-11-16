// Function: sub_39CB2D0
// Address: 0x39cb2d0
//
void __fastcall sub_39CB2D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  __int64 *v8; // r8
  __int64 v9; // rdx
  __int64 *v10; // r13
  __int64 *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rax
  char *v14; // rax
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // [rsp+8h] [rbp-A8h]
  char *v18; // [rsp+20h] [rbp-90h] BYREF
  __int64 v19; // [rsp+28h] [rbp-88h]
  _BYTE v20[32]; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v21[2]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE v22[80]; // [rsp+60h] [rbp-50h] BYREF

  v7 = *(unsigned int *)(a3 + 8);
  v18 = v20;
  v19 = 0x200000000LL;
  if ( (unsigned int)v7 <= 2 )
  {
    v8 = *(__int64 **)a3;
    v9 = 16 * v7;
    v10 = (__int64 *)(*(_QWORD *)a3 + v9);
    if ( v10 != *(__int64 **)a3 )
      goto LABEL_3;
LABEL_15:
    v15 = v19;
    goto LABEL_7;
  }
  sub_16CD150((__int64)&v18, v20, v7, 16, a5, a6);
  v8 = *(__int64 **)a3;
  v9 = 16LL * *(unsigned int *)(a3 + 8);
  v10 = (__int64 *)(*(_QWORD *)a3 + v9);
  if ( v10 == *(__int64 **)a3 )
    goto LABEL_15;
LABEL_3:
  v11 = v8;
  do
  {
    v12 = sub_397FB50(a1[25], v11[1]);
    a6 = sub_397FAE0(a1[25], *v11);
    v13 = (unsigned int)v19;
    if ( (unsigned int)v19 >= HIDWORD(v19) )
    {
      v17 = a6;
      sub_16CD150((__int64)&v18, v20, 0, 16, (int)v8, a6);
      v13 = (unsigned int)v19;
      a6 = v17;
    }
    v14 = &v18[16 * v13];
    v11 += 2;
    *(_QWORD *)v14 = a6;
    *((_QWORD *)v14 + 1) = v12;
    v15 = v19 + 1;
    LODWORD(v19) = v19 + 1;
  }
  while ( v10 != v11 );
LABEL_7:
  v16 = 0x200000000LL;
  v21[0] = (unsigned __int64)v22;
  v21[1] = 0x200000000LL;
  if ( v15 )
    sub_39C75B0((__int64)v21, &v18, v9, 0x200000000LL, (int)v8, a6);
  sub_39CB220(a1, a2, (__int64)v21, v16, (int)v8, a6);
  if ( (_BYTE *)v21[0] != v22 )
    _libc_free(v21[0]);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
}
