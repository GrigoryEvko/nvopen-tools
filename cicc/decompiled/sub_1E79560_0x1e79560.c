// Function: sub_1E79560
// Address: 0x1e79560
//
void __fastcall sub_1E79560(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  unsigned __int64 *v15; // rbx
  __int64 *v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 *v20; // r14
  __int64 *v21; // r13
  unsigned __int64 *v22; // rbx
  __int64 v23; // rax
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 i; // r8
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v35; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v36; // [rsp+30h] [rbp-50h] BYREF
  __int64 v37; // [rsp+38h] [rbp-48h]
  _BYTE v38[64]; // [rsp+40h] [rbp-40h] BYREF

  v36 = (__int64 *)v38;
  v37 = 0x200000000LL;
  if ( !**(_BYTE **)(a1 + 32) )
  {
    v28 = a1;
    if ( (*(_BYTE *)a1 & 4) == 0 && (*(_BYTE *)(a1 + 46) & 8) != 0 )
    {
      do
        v28 = *(_QWORD *)(v28 + 8);
      while ( (*(_BYTE *)(v28 + 46) & 8) != 0 );
    }
    v29 = *(_QWORD *)(v28 + 8);
    for ( i = *(_QWORD *)(a1 + 24) + 24LL; v29 != i; v29 = *(_QWORD *)(v29 + 8) )
    {
      if ( **(_WORD **)(v29 + 16) != 12 )
        break;
      v31 = *(_QWORD *)(v29 + 32);
      if ( !*(_BYTE *)v31 && *(_DWORD *)(v31 + 8) == *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL) )
      {
        v32 = (unsigned int)v37;
        if ( (unsigned int)v37 >= HIDWORD(v37) )
        {
          v34 = i;
          sub_16CD150((__int64)&v36, v38, 0, 8, i, a6);
          v32 = (unsigned int)v37;
          i = v34;
        }
        v36[v32] = v29;
        LODWORD(v37) = v37 + 1;
      }
      if ( (*(_BYTE *)v29 & 4) == 0 && (*(_BYTE *)(v29 + 46) & 8) != 0 )
      {
        do
          v29 = *(_QWORD *)(v29 + 8);
        while ( (*(_BYTE *)(v29 + 46) & 8) != 0 );
      }
    }
  }
  if ( a2 + 24 == (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) || (unsigned __int64 *)(a2 + 24) == a3 )
  {
    v27 = *(_QWORD *)(a1 + 64);
    v35 = 0;
    if ( v27 )
    {
      sub_161E7C0(a1 + 64, v27);
      *(_QWORD *)(a1 + 64) = v35;
    }
  }
  else
  {
    v9 = sub_15C70A0((__int64)(a3 + 8));
    v10 = sub_15C70A0(a1 + 64);
    v11 = sub_15BA070(v10, v9, 0);
    sub_15C7080(&v35, v11);
    v12 = *(_QWORD *)(a1 + 64);
    if ( v12 )
      sub_161E7C0(a1 + 64, v12);
    v13 = v35;
    *(_QWORD *)(a1 + 64) = v35;
    if ( v13 )
      sub_1623210((__int64)&v35, v13, a1 + 64);
  }
  v14 = a1;
  if ( (*(_BYTE *)a1 & 4) == 0 && (*(_BYTE *)(a1 + 46) & 8) != 0 )
  {
    do
      v14 = *(_QWORD *)(v14 + 8);
    while ( (*(_BYTE *)(v14 + 46) & 8) != 0 );
  }
  v15 = *(unsigned __int64 **)(v14 + 8);
  v16 = (__int64 *)(a2 + 16);
  v17 = *(_QWORD *)(a1 + 24) + 16LL;
  if ( (unsigned __int64 *)a1 != v15 && a3 != v15 )
  {
    if ( v16 != (__int64 *)v17 )
    {
      v33 = *(_QWORD *)(a1 + 24) + 16LL;
      sub_1DD5C00(v16, v17, a1, *(_QWORD *)(v14 + 8));
      v17 = v33;
      v16 = (__int64 *)(a2 + 16);
    }
    if ( v15 != a3 && v15 != (unsigned __int64 *)a1 )
    {
      v18 = *v15 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v15;
      *v15 = *v15 & 7 | *(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = *a3;
      *(_QWORD *)(v18 + 8) = a3;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)a1 = v19 | *(_QWORD *)a1 & 7LL;
      *(_QWORD *)(v19 + 8) = a1;
      *a3 = v18 | *a3 & 7;
    }
  }
  v20 = v36;
  v21 = &v36[(unsigned int)v37];
  if ( v36 != v21 )
  {
    do
    {
      v22 = (unsigned __int64 *)*v20;
      if ( !*v20 )
        BUG();
      v23 = *v20;
      if ( (*(_BYTE *)v22 & 4) == 0 && (*((_BYTE *)v22 + 46) & 8) != 0 )
      {
        do
          v23 = *(_QWORD *)(v23 + 8);
        while ( (*(_BYTE *)(v23 + 46) & 8) != 0 );
      }
      v24 = *(unsigned __int64 **)(v23 + 8);
      if ( v22 != v24 && a3 != v24 )
      {
        if ( v16 != (__int64 *)v17 )
          sub_1DD5C00(v16, v17, *v20, *(_QWORD *)(v23 + 8));
        if ( v24 != a3 && v24 != v22 )
        {
          v25 = *v24 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v22 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v24;
          *v24 = *v24 & 7 | *v22 & 0xFFFFFFFFFFFFFFF8LL;
          v26 = *a3;
          *(_QWORD *)(v25 + 8) = a3;
          v26 &= 0xFFFFFFFFFFFFFFF8LL;
          *v22 = v26 | *v22 & 7;
          *(_QWORD *)(v26 + 8) = v22;
          *a3 = v25 | *a3 & 7;
        }
      }
      ++v20;
    }
    while ( v21 != v20 );
    v21 = v36;
  }
  if ( v21 != (__int64 *)v38 )
    _libc_free((unsigned __int64)v21);
}
