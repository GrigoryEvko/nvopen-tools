// Function: sub_F26350
// Address: 0xf26350
//
unsigned __int8 *__fastcall sub_F26350(__int64 a1, _BYTE *a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rbx
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  const char *v35[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v36; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a3 + 16);
  if ( (!v6 || *(_QWORD *)(v6 + 8)) && !a4 )
    return 0;
  v7 = *(_QWORD *)(a3 + 8);
  v33 = *(_QWORD *)(a3 - 64);
  v32 = *(_QWORD *)(a3 - 32);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  if ( sub_BCAC40(v7, 1) )
    return 0;
  v11 = *(_QWORD *)(a3 - 96);
  if ( *(_BYTE *)v11 == 83 )
  {
    v27 = *(_QWORD *)(v11 + 16);
    if ( v27 )
    {
      if ( !*(_QWORD *)(v27 + 8) )
      {
        v28 = *(_QWORD *)(v11 - 64);
        v29 = *(_QWORD *)(v11 - 32);
        if ( v33 == v28 && v32 == v29 )
          return 0;
        if ( v32 == v28 && v33 == v29 )
          return 0;
      }
    }
  }
  v12 = sub_F07F50((__int64)a2, (unsigned __int8 *)a3, 1, v8, v9, v10);
  v16 = sub_F07F50((__int64)a2, (unsigned __int8 *)a3, 0, v13, v14, v15);
  if ( !(v16 | v12) )
    return 0;
  if ( v12 )
  {
    if ( !v16 )
    {
      v16 = sub_B47F80(a2);
      sub_BD2ED0(v16, a3, v32);
      sub_B44E20((unsigned __int8 *)v16);
      sub_B44220((_QWORD *)v16, (__int64)(a2 + 24), 0);
      v31 = *(_QWORD *)(a1 + 40);
      v35[0] = (const char *)v16;
      sub_F200C0(v31 + 2096, (__int64 *)v35);
    }
  }
  else
  {
    v12 = sub_B47F80(a2);
    sub_BD2ED0(v12, a3, v33);
    sub_B44E20((unsigned __int8 *)v12);
    sub_B44220((_QWORD *)v12, (__int64)(a2 + 24), 0);
    v30 = *(_QWORD *)(a1 + 40);
    v35[0] = (const char *)v12;
    sub_F200C0(v30 + 2096, (__int64 *)v35);
  }
  v17 = *(_QWORD *)(a3 - 96);
  v36 = 257;
  v18 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v19 = v18;
  if ( v18 )
  {
    v34 = (__int64)v18;
    sub_B44260((__int64)v18, *(_QWORD *)(v12 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v19 - 12) )
    {
      v20 = *((_QWORD *)v19 - 11);
      **((_QWORD **)v19 - 10) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *((_QWORD *)v19 - 10);
    }
    *((_QWORD *)v19 - 12) = v17;
    if ( v17 )
    {
      v21 = *(_QWORD *)(v17 + 16);
      *((_QWORD *)v19 - 11) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = v19 - 88;
      *((_QWORD *)v19 - 10) = v17 + 16;
      *(_QWORD *)(v17 + 16) = v19 - 96;
    }
    if ( *((_QWORD *)v19 - 8) )
    {
      v22 = *((_QWORD *)v19 - 7);
      **((_QWORD **)v19 - 6) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = *((_QWORD *)v19 - 6);
    }
    *((_QWORD *)v19 - 8) = v12;
    v23 = *(_QWORD *)(v12 + 16);
    *((_QWORD *)v19 - 7) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = v19 - 56;
    *((_QWORD *)v19 - 6) = v12 + 16;
    *(_QWORD *)(v12 + 16) = v19 - 64;
    if ( *((_QWORD *)v19 - 4) )
    {
      v24 = *((_QWORD *)v19 - 3);
      **((_QWORD **)v19 - 2) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = *((_QWORD *)v19 - 2);
    }
    *((_QWORD *)v19 - 4) = v16;
    if ( v16 )
    {
      v25 = *(_QWORD *)(v16 + 16);
      *((_QWORD *)v19 - 3) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = v19 - 24;
      *((_QWORD *)v19 - 2) = v16 + 16;
      *(_QWORD *)(v16 + 16) = v19 - 32;
    }
    sub_BD6B50(v19, v35);
    sub_B47C00(v34, a3, 0, 0);
  }
  else
  {
    sub_B47C00(0, a3, 0, 0);
  }
  return v19;
}
