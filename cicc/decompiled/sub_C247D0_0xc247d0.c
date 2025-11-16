// Function: sub_C247D0
// Address: 0xc247d0
//
__int64 __fastcall sub_C247D0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rax
  _QWORD *v7; // r15
  unsigned __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int64 v13; // rdi
  char v14; // al
  unsigned __int64 v15; // r9
  unsigned __int64 *v16; // rax
  unsigned __int64 *v17; // rcx
  __int64 v18; // rax
  unsigned __int64 *v19; // rcx
  __int64 v20; // rax
  __int64 (__fastcall ***v21)(); // rax
  unsigned __int64 *v22; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v23; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v24; // [rsp+20h] [rbp-90h]
  unsigned __int64 v26; // [rsp+28h] [rbp-88h]
  unsigned __int64 *v27; // [rsp+28h] [rbp-88h]
  unsigned __int64 v28; // [rsp+28h] [rbp-88h]
  unsigned __int64 v29; // [rsp+28h] [rbp-88h]
  __int64 v30; // [rsp+30h] [rbp-80h] BYREF
  __int64 v31; // [rsp+38h] [rbp-78h] BYREF
  __int64 v32; // [rsp+40h] [rbp-70h] BYREF
  __int64 v33; // [rsp+48h] [rbp-68h] BYREF
  __int64 v34; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v35; // [rsp+58h] [rbp-58h] BYREF
  __int64 v36; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v37; // [rsp+68h] [rbp-48h] BYREF
  __int64 v38[8]; // [rsp+70h] [rbp-40h] BYREF

  v6 = (_QWORD *)sub_22077B0(8);
  v7 = v6;
  if ( v6 )
  {
    *v6 = 0;
    sub_EE5C10(v6);
  }
  sub_C2DFC0(&v30, v7, *a2);
  v8 = v30 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v38[0] = a4;
    v13 = v30 & 0xFFFFFFFFFFFFFFFELL;
    v38[1] = (__int64)a2;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v26 = v8;
    v14 = (*(__int64 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v8 + 48LL))(v13, &unk_4F84052);
    v15 = v26;
    if ( v14 )
    {
      v16 = *(unsigned __int64 **)(v26 + 16);
      v17 = *(unsigned __int64 **)(v26 + 8);
      v33 = 1;
      v22 = v16;
      if ( v17 == v16 )
      {
        v20 = 1;
      }
      else
      {
        do
        {
          v23 = v15;
          v27 = v17;
          v35 = *v17;
          *v17 = 0;
          sub_C1F7B0(&v36, &v35, v38);
          v18 = v33;
          v33 = 0;
          v34 = v18 | 1;
          sub_9CDB40(&v37, (unsigned __int64 *)&v34, (unsigned __int64 *)&v36);
          if ( (v33 & 1) != 0 || (v19 = v27, v15 = v23, (v33 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_C63C30(&v33);
          v33 |= v37 | 1;
          if ( (v34 & 1) != 0 || (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v34);
          if ( (v36 & 1) != 0 || (v36 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v36);
          if ( v35 )
          {
            v24 = v27;
            v28 = v15;
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v35 + 8LL))(v35);
            v19 = v24;
            v15 = v28;
          }
          v17 = v19 + 1;
        }
        while ( v22 != v17 );
        v20 = v33 | 1;
      }
      v29 = v15;
      v36 = v20;
      v33 = 0;
      sub_9C66B0(&v33);
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v29 + 8LL))(v29);
    }
    else
    {
      v37 = v26;
      sub_C1F7B0(&v36, &v37, v38);
      if ( v37 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v37 + 8LL))(v37);
    }
    if ( (v36 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v32 & 1) != 0 || (v32 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v32);
    if ( (v31 & 1) != 0 || (v31 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v31);
    *(_BYTE *)(a1 + 16) |= 1u;
    v21 = sub_C1AFD0();
    *(_DWORD *)a1 = 5;
    *(_QWORD *)(a1 + 8) = v21;
    if ( (v30 & 1) != 0 || (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v30);
    if ( v7 )
    {
      sub_EE5E50(v7);
      j_j___libc_free_0(v7, 8);
    }
  }
  else
  {
    v9 = *a2;
    *a2 = 0;
    v10 = sub_22077B0(64);
    v11 = v10;
    if ( v10 )
    {
      *(_QWORD *)v10 = v9;
      *(_QWORD *)(v10 + 8) = v7;
      *(_QWORD *)(v10 + 16) = 0;
      *(_QWORD *)(v10 + 24) = 0;
      *(_QWORD *)(v10 + 32) = 0;
      *(_DWORD *)(v10 + 40) = 0;
      *(_BYTE *)(v10 + 56) = 0;
      *(_QWORD *)(v10 + 48) = a3;
    }
    else
    {
      if ( v7 )
      {
        sub_EE5E50(v7);
        j_j___libc_free_0(v7, 8);
      }
      if ( v9 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
    }
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v11;
  }
  return a1;
}
