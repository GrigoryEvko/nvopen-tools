// Function: sub_22E4210
// Address: 0x22e4210
//
__int64 __fastcall sub_22E4210(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 i; // rsi
  __int64 v11; // rdx
  __int64 *v12; // r13
  __int64 v13; // r12
  unsigned int v14; // ebx
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __int64 *v19; // r12
  __int64 v20; // r13
  _QWORD *v21; // rax
  _QWORD *v22; // r8
  __int64 (__fastcall *v23)(__int64, __int64 *); // rax
  char *v24; // rax
  __int64 v25; // rdx
  bool v26; // al
  __int64 v27; // r8
  bool v28; // r13
  _QWORD *v29; // rax
  char v30; // al
  __int64 v31; // rax
  _QWORD *v32; // rdi
  int v33; // r13d
  unsigned int v34; // ebx
  __int64 (*v35)(); // rax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int8 v40; // [rsp+7h] [rbp-89h]
  __int64 *v41; // [rsp+8h] [rbp-88h]
  _QWORD *v42; // [rsp+10h] [rbp-80h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  __int64 v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 *v46; // [rsp+20h] [rbp-70h]
  __int64 *v47; // [rsp+28h] [rbp-68h]
  unsigned int v48; // [rsp+28h] [rbp-68h]
  __m128i v49; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v50[10]; // [rsp+40h] [rbp-50h] BYREF

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_63:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4FDBD0C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_63;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4FDBD0C);
  v6 = *(_QWORD *)(a1 + 184);
  v7 = a1 + 336;
  v8 = v5;
  *(_QWORD *)(a1 + 648) = v5 + 176;
  v9 = *(_QWORD *)(v6 + 16);
  for ( i = *(_QWORD *)(v6 + 8); v9 != i; *(_QWORD *)(v7 - 8) = v11 + 208 )
  {
    v11 = *(_QWORD *)(v9 - 8);
    v9 -= 8;
    v7 += 8;
  }
  sub_22E2DD0(*(_QWORD *)(v8 + 208), (unsigned __int64 *)(a1 + 568));
  v12 = *(__int64 **)(a1 + 584);
  v46 = *(__int64 **)(a1 + 616);
  if ( v12 == v46 )
  {
    return 0;
  }
  else
  {
    v40 = 0;
    v47 = *(__int64 **)(a1 + 600);
    v45 = *(_QWORD *)(a1 + 608);
    do
    {
      v13 = *v12;
      if ( *(_DWORD *)(a1 + 200) )
      {
        v14 = 0;
        do
        {
          while ( 1 )
          {
            v15 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v14);
            v16 = *(__int64 (**)())(*(_QWORD *)v15 + 152LL);
            if ( v16 != sub_22E2A30 )
              break;
            if ( ++v14 >= *(_DWORD *)(a1 + 200) )
              goto LABEL_14;
          }
          ++v14;
          v40 |= ((__int64 (__fastcall *)(__int64, __int64, __int64))v16)(v15, v13, a1);
        }
        while ( v14 < *(_DWORD *)(a1 + 200) );
      }
LABEL_14:
      if ( v47 == ++v12 )
      {
        v12 = *(__int64 **)(v45 + 8);
        v45 += 8;
        v47 = v12 + 64;
      }
    }
    while ( v46 != v12 );
LABEL_17:
    v17 = *(_QWORD *)(a1 + 616);
    if ( v17 != *(_QWORD *)(a1 + 584) )
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(a1 + 624);
        if ( v18 == v17 )
          v17 = *(_QWORD *)(*(_QWORD *)(a1 + 640) - 8LL) + 512LL;
        *(_QWORD *)(a1 + 656) = *(_QWORD *)(v17 - 8);
        if ( *(_DWORD *)(a1 + 200) )
          break;
LABEL_50:
        v31 = *(_QWORD *)(a1 + 616);
        if ( v31 == v18 )
        {
          j_j___libc_free_0(v18);
          v37 = (__int64 *)(*(_QWORD *)(a1 + 640) - 8LL);
          *(_QWORD *)(a1 + 640) = v37;
          v38 = *v37;
          v39 = *v37 + 512;
          *(_QWORD *)(a1 + 624) = v38;
          *(_QWORD *)(a1 + 632) = v39;
          *(_QWORD *)(a1 + 616) = v38 + 504;
        }
        else
        {
          *(_QWORD *)(a1 + 616) = v31 - 8;
        }
        v32 = *(_QWORD **)(*(_QWORD *)(a1 + 648) + 32LL);
        if ( !v32 )
          goto LABEL_17;
        sub_22DB850(v32);
        v17 = *(_QWORD *)(a1 + 616);
        if ( v17 == *(_QWORD *)(a1 + 584) )
          goto LABEL_54;
      }
      v48 = 0;
      while ( 1 )
      {
        v19 = *(__int64 **)(*(_QWORD *)(a1 + 192) + 8LL * v48);
        if ( sub_B80690() )
        {
          sub_22DAE20(&v49, *(__int64 **)(a1 + 656));
          sub_B817B0(a1 + 176, (__int64)v19, 0, 5, (const void *)v49.m128i_i64[0], v49.m128i_u64[1]);
          if ( (_QWORD *)v49.m128i_i64[0] != v50 )
            j_j___libc_free_0(v49.m128i_u64[0]);
          sub_B86470(a1 + 176, v19);
        }
        sub_B89740(a1 + 176, v19);
        v20 = **(_QWORD **)(a1 + 656);
        sub_C85EE0(&v49);
        v50[0] = v19;
        v50[2] = 0;
        v49.m128i_i64[0] = (__int64)&unk_49DA748;
        v50[1] = v20 & 0xFFFFFFFFFFFFFFF8LL;
        v21 = sub_BC4450(v19);
        v22 = v21;
        if ( v21 )
        {
          v42 = v21;
          sub_C9E250((__int64)v21);
          v22 = v42;
        }
        v43 = (__int64)v22;
        v23 = *(__int64 (__fastcall **)(__int64, __int64 *))(*v19 + 144);
        if ( v23 == sub_22E41D0 )
        {
          v41 = *(__int64 **)(a1 + 656);
          v24 = (char *)sub_BD5D20(*(_QWORD *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 72));
          v26 = sub_BC63A0(v24, v25);
          v27 = v43;
          v28 = v26;
          if ( v26 )
          {
            v28 = 0;
            sub_22E39A0((__int64)v19, v41);
            v27 = v43;
          }
        }
        else
        {
          v30 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v23)(v19, *(_QWORD *)(a1 + 656), a1);
          v27 = v43;
          v40 |= v30;
          v28 = v30;
        }
        if ( v27 )
          sub_C9E2A0(v27);
        v49.m128i_i64[0] = (__int64)&unk_49DA748;
        nullsub_162();
        if ( sub_B80690() )
        {
          if ( v28 )
          {
            sub_22DAE20(&v49, *(__int64 **)(a1 + 656));
            sub_B817B0(a1 + 176, (__int64)v19, 1, 5, (const void *)v49.m128i_i64[0], v49.m128i_u64[1]);
            if ( (_QWORD *)v49.m128i_i64[0] != v50 )
              j_j___libc_free_0(v49.m128i_u64[0]);
          }
          sub_B865A0(a1 + 176, v19);
        }
        v29 = sub_BC4450(v19);
        if ( v29 )
        {
          v44 = (__int64)v29;
          sub_C9E250((__int64)v29);
          sub_22DCAC0(*(__int64 **)(a1 + 656));
          sub_C9E2A0(v44);
          nullsub_76();
          if ( v28 )
            goto LABEL_42;
        }
        else
        {
          sub_22DCAC0(*(__int64 **)(a1 + 656));
          nullsub_76();
          if ( v28 )
LABEL_42:
            sub_B887D0(a1 + 176, v19);
        }
        sub_B87180(a1 + 176, (__int64)v19);
        if ( sub_B80690() )
        {
          sub_22DAE20(&v49, *(__int64 **)(a1 + 656));
        }
        else
        {
          v49.m128i_i64[0] = (__int64)v50;
          sub_22E2D20(v49.m128i_i64, "<deleted>", (__int64)"");
        }
        sub_B81BF0(a1 + 176, (__int64)v19, (const void *)v49.m128i_i64[0], v49.m128i_u64[1], 5);
        if ( (_QWORD *)v49.m128i_i64[0] != v50 )
          j_j___libc_free_0(v49.m128i_u64[0]);
        if ( ++v48 >= *(_DWORD *)(a1 + 200) )
        {
          v18 = *(_QWORD *)(a1 + 624);
          goto LABEL_50;
        }
      }
    }
LABEL_54:
    if ( *(_DWORD *)(a1 + 200) )
    {
      v33 = v40;
      v34 = 0;
      do
      {
        while ( 1 )
        {
          v35 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 192) + 8LL * v34) + 160LL);
          if ( v35 != sub_22E2A40 )
            break;
          if ( ++v34 >= *(_DWORD *)(a1 + 200) )
            return (unsigned __int8)v33;
        }
        ++v34;
        v33 |= v35();
      }
      while ( v34 < *(_DWORD *)(a1 + 200) );
      return (unsigned __int8)v33;
    }
  }
  return v40;
}
