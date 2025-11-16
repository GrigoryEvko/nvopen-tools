// Function: sub_2357CF0
// Address: 0x2357cf0
//
void __fastcall sub_2357CF0(unsigned __int64 *a1, _QWORD *a2)
{
  _QWORD *v3; // rdx
  __int64 v4; // rcx
  _QWORD *v5; // rax
  _QWORD *v6; // rcx
  _QWORD *v7; // rsi
  __int64 v8; // rdi
  _QWORD *v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rbx
  unsigned __int64 v12; // r12
  __int64 v13; // rdi
  _QWORD *v14; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v15; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-38h]
  __int64 v17; // [rsp+20h] [rbp-30h]

  v3 = (_QWORD *)a2[1];
  v4 = a2[2];
  v15 = (_QWORD *)*a2;
  v16 = v3;
  v17 = v4;
  if ( a2 == v15 )
  {
    v16 = &v15;
    v15 = &v15;
  }
  else
  {
    *v3 = &v15;
    v15[1] = &v15;
    a2[1] = a2;
    *a2 = a2;
    a2[2] = 0;
  }
  v5 = (_QWORD *)sub_22077B0(0x20u);
  if ( v5 )
  {
    v6 = v15;
    v7 = v16;
    v8 = v17;
    v5[1] = v15;
    *v5 = &unk_4A0E078;
    v9 = v5 + 1;
    v5[2] = v7;
    v5[3] = v8;
    if ( v6 == &v15 )
    {
      v5[2] = v9;
      v5[1] = v9;
    }
    else
    {
      *v7 = v9;
      v10 = v5[1];
      v15 = &v15;
      *(_QWORD *)(v10 + 8) = v9;
      v16 = &v15;
      v17 = 0;
    }
  }
  v14 = v5;
  sub_2356EF0(a1, (unsigned __int64 *)&v14);
  if ( v14 )
    (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
  v11 = v15;
  while ( v11 != &v15 )
  {
    v12 = (unsigned __int64)v11;
    v11 = (_QWORD *)*v11;
    v13 = *(_QWORD *)(v12 + 16);
    if ( v13 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
    j_j___libc_free_0(v12);
  }
}
