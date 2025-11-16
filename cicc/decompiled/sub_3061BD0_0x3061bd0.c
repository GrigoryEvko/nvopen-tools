// Function: sub_3061BD0
// Address: 0x3061bd0
//
void __fastcall sub_3061BD0(__int64 *a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  _QWORD *v6; // r13
  char v7; // bl
  __int64 v8; // rax
  __int64 v9; // rdi
  char *v10; // rsi
  char *v11; // rbx
  char *v12; // r12
  __int64 v13; // [rsp+8h] [rbp-68h] BYREF
  _QWORD *v14; // [rsp+10h] [rbp-60h] BYREF
  char v15; // [rsp+18h] [rbp-58h]
  char *v16; // [rsp+20h] [rbp-50h] BYREF
  char *v17; // [rsp+28h] [rbp-48h]
  char *v18; // [rsp+30h] [rbp-40h]
  __int64 v19; // [rsp+38h] [rbp-38h]
  __int64 v20; // [rsp+40h] [rbp-30h]

  v3 = *a1;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v4 = (_QWORD *)sub_22077B0(0x10u);
  v5 = v4;
  if ( v4 )
  {
    v4[1] = v3;
    *v4 = &unk_4A0FBB8;
  }
  v14 = v4;
  if ( v17 == v18 )
  {
    sub_2353750((unsigned __int64 *)&v16, v17, &v14);
    v5 = v14;
  }
  else
  {
    if ( v17 )
    {
      *(_QWORD *)v17 = v4;
      v17 += 8;
      goto LABEL_6;
    }
    v17 = (char *)8;
  }
  if ( v5 )
    (*(void (__fastcall **)(_QWORD *))(*v5 + 8LL))(v5);
LABEL_6:
  sub_234AAB0((__int64)&v14, (__int64 *)&v16, 0);
  v6 = v14;
  v7 = v15;
  v14 = 0;
  v8 = sub_22077B0(0x18u);
  v9 = v8;
  if ( v8 )
  {
    *(_BYTE *)(v8 + 16) = v7;
    *(_QWORD *)(v8 + 8) = v6;
    v6 = 0;
    *(_QWORD *)v8 = &unk_4A0C478;
  }
  v13 = v8;
  v10 = (char *)a2[1];
  if ( v10 == (char *)a2[2] )
  {
    sub_2275C60(a2, v10, &v13);
    v9 = v13;
  }
  else
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = v8;
      a2[1] += 8LL;
      goto LABEL_11;
    }
    a2[1] = 8;
  }
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
LABEL_11:
  if ( v6 )
    (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
  if ( v14 )
    (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
  v11 = v17;
  v12 = v16;
  if ( v17 != v16 )
  {
    do
    {
      if ( *(_QWORD *)v12 )
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v12 + 8LL))(*(_QWORD *)v12);
      v12 += 8;
    }
    while ( v11 != v12 );
    v12 = v16;
  }
  if ( v12 )
    j_j___libc_free_0((unsigned __int64)v12);
}
