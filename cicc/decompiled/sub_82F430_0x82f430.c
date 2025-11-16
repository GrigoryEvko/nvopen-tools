// Function: sub_82F430
// Address: 0x82f430
//
void __fastcall sub_82F430(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 *a4,
        int a5,
        unsigned int a6,
        int a7,
        int a8,
        __m128i *a9,
        int *a10,
        int a11)
{
  __int64 v13; // r13
  char v15; // al
  char v16; // al
  __int64 v17; // rcx
  char v18; // cl
  char v19; // cl
  __int64 *v20; // rbx
  int v21; // edx
  char v22; // al
  __int64 *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  char j; // dl
  __int64 i; // rax
  unsigned __int64 *v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // [rsp-10h] [rbp-B0h]
  _DWORD *v36; // [rsp+0h] [rbp-A0h]
  int v37; // [rsp+Ch] [rbp-94h]
  _BOOL4 v38; // [rsp+10h] [rbp-90h]
  _BOOL4 v39; // [rsp+14h] [rbp-8Ch]
  __int64 *v40; // [rsp+20h] [rbp-80h]
  __m128i v43; // [rsp+30h] [rbp-70h] BYREF
  char v44; // [rsp+41h] [rbp-5Fh]

  v13 = a1;
  v15 = *(_BYTE *)(a1 + 80);
  if ( v15 == 16 )
  {
    v13 = **(_QWORD **)(a1 + 88);
    v15 = *(_BYTE *)(v13 + 80);
  }
  if ( v15 == 24 )
    v13 = *(_QWORD *)(v13 + 88);
  v16 = *(_BYTE *)(a2 + 80);
  v17 = a2;
  if ( v16 == 16 )
  {
    v17 = **(_QWORD **)(a2 + 88);
    v16 = *(_BYTE *)(v17 + 80);
  }
  if ( v16 == 24 )
    v16 = *(_BYTE *)(*(_QWORD *)(v17 + 88) + 80LL);
  if ( a3 )
  {
    v18 = a3[19];
    v39 = (a3[18] & 0x40) != 0;
    if ( (v18 & 2) != 0 )
    {
      v36 = a3 + 120;
      v38 = (v18 & 4) != 0;
    }
    else
    {
      v36 = 0;
      v38 = 0;
    }
    v19 = a3[16];
    a4 = (__int64 *)(a3 + 68);
    v40 = (__int64 *)(a3 + 76);
    if ( v19 == 3 || (v20 = (__int64 *)(a3 + 68), v19 == 6) )
      v20 = (__int64 *)(a3 + 112);
  }
  else
  {
    v40 = a4;
    v20 = a4;
    v36 = 0;
    v38 = 0;
    v39 = 0;
  }
  v21 = 0;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
    v21 = a8;
  v37 = v21;
  if ( v16 == 17 || v16 == 20 )
  {
    sub_878710(a1, &v43);
    v43.m128i_i64[1] = *v20;
    sub_6E6370(&v43, a2);
  }
  else
  {
    sub_878710(a2, &v43);
    v43.m128i_i64[1] = *v20;
    if ( a11 | a6 || *(char *)(qword_4D03C50 + 18LL) < 0 )
      sub_6E62A0((__int64)&v43);
  }
  v22 = v44;
  *a10 = v44 & 1;
  if ( (v22 & 0x20) != 0 )
  {
    if ( a9 )
    {
      sub_6E6260(a9);
      *(__int64 *)((char *)a9[4].m128i_i64 + 4) = *a4;
      *(__int64 *)((char *)&a9[4].m128i_i64[1] + 4) = *v40;
    }
  }
  else if ( dword_4F07734
         && (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0
         && (*(_BYTE *)(*(_QWORD *)(v13 + 88) + 206LL) & 0x10) != 0 )
  {
    sub_6E50A0();
    if ( a9 )
    {
      sub_6E6260(a9);
      *(__int64 *)((char *)a9[4].m128i_i64 + 4) = *a4;
    }
  }
  else if ( a5 )
  {
    sub_8767A0(4, v13, v20, 0);
  }
  else if ( a9 )
  {
    if ( a8 && *(_BYTE *)(v13 + 80) == 10 )
    {
      for ( i = *(_QWORD *)(*(_QWORD *)(v13 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v23 = v20;
      v24 = v13;
      if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
      {
        v31 = (unsigned __int64 *)sub_6E50B0(v13, v20);
        sub_6EA7D0(a1, a2, a4, v40, *a10 == 0, v39, v38, 0, a9);
        a9[5].m128i_i64[1] = (__int64)v31;
        sub_6E5820(v31, 32);
        return;
      }
    }
    else
    {
      v23 = v20;
      v24 = v13;
    }
    v25 = sub_6E50B0(v24, v23);
    sub_6EAB60(a1, v39, a6, a4, v40, v25, (__int64)a9);
    if ( !a7 && a9[1].m128i_i8[0] )
    {
      v28 = a9->m128i_i64[0];
      for ( j = *(_BYTE *)(a9->m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v28 + 140) )
        v28 = *(_QWORD *)(v28 + 160);
      if ( j )
      {
        sub_6F5960(a9, a8 == 0, v36, v35, v26, v27);
        if ( v37 )
          sub_6E5A30(a9[5].m128i_i64[1], 4, 32);
      }
    }
  }
  else
  {
    sub_8767A0((-(__int64)(v37 == 0) & 0xFFFFFFFFFFFFFFE0LL) + 36, v13, v20, 0);
    sub_6E1D20(*(__int64 **)(v13 + 88), v13, v32, v33, v34);
  }
}
