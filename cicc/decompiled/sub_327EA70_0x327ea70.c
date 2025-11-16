// Function: sub_327EA70
// Address: 0x327ea70
//
__int64 __fastcall sub_327EA70(_QWORD *a1, __int64 a2)
{
  unsigned int *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // rax
  __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // r15
  bool v10; // zf
  void *v12; // r14
  void *v13; // rax
  void *v14; // rbx
  _QWORD *i; // rbx
  unsigned int v16; // [rsp+0h] [rbp-70h] BYREF
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+10h] [rbp-60h] BYREF
  int v19; // [rsp+18h] [rbp-58h]
  void *v20; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v21; // [rsp+28h] [rbp-48h]

  v3 = *(unsigned int **)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)v3;
  v6 = *(_QWORD *)(*(_QWORD *)v3 + 48LL) + 16LL * v3[2];
  v7 = *(_WORD *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  v18 = v4;
  LOWORD(v16) = v7;
  v17 = v8;
  if ( v4 )
    sub_B96E90((__int64)&v18, v4, 1);
  v9 = 0;
  v10 = *(_DWORD *)(v5 + 24) == 51;
  v19 = *(_DWORD *)(a2 + 72);
  if ( v10 )
  {
    v12 = sub_300AC80((unsigned __int16 *)&v16, v4);
    v13 = sub_C33340();
    v14 = v13;
    if ( v12 == v13 )
      sub_C3C500(&v20, (__int64)v13);
    else
      sub_C373C0(&v20, (__int64)v12);
    if ( v20 == v14 )
      sub_C3D480((__int64)&v20, 0, 0, 0);
    else
      sub_C36070((__int64)&v20, 0, 0, 0);
    v9 = sub_33FE6E0(*a1, &v20, &v18, v16, v17, 0);
    if ( v20 == v14 )
    {
      if ( v21 )
      {
        for ( i = &v21[3 * *(v21 - 1)]; v21 != i; sub_91D830(i) )
          i -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v20);
    }
  }
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v9;
}
