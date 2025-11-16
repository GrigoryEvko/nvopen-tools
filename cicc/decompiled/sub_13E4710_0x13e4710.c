// Function: sub_13E4710
// Address: 0x13e4710
//
__int64 __fastcall sub_13E4710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 i; // r14
  __int64 v9; // rax
  __int64 v10; // r8
  int v11; // eax
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r14d
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // r13
  int v23; // eax
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v34; // [rsp+20h] [rbp-110h]
  __int64 v35; // [rsp+28h] [rbp-108h] BYREF
  __m128i v36; // [rsp+30h] [rbp-100h] BYREF
  __int64 v37; // [rsp+40h] [rbp-F0h]
  __int64 v38; // [rsp+48h] [rbp-E8h]
  __int64 v39; // [rsp+50h] [rbp-E0h]
  __int64 v40; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v41; // [rsp+68h] [rbp-C8h]
  __int64 v42; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE *v43; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v44; // [rsp+B8h] [rbp-78h]
  _BYTE v45[112]; // [rsp+C0h] [rbp-70h] BYREF

  v5 = (unsigned __int64 *)&v42;
  v35 = a1;
  v40 = 0;
  v41 = 1;
  do
    *v5++ = -8;
  while ( v5 != (unsigned __int64 *)&v43 );
  v43 = v45;
  v44 = 0x800000000LL;
  v6 = sub_15F2050(v35);
  v34 = sub_1632FA0(v6);
  if ( a2 )
  {
    v7 = v35;
    for ( i = *(_QWORD *)(v35 + 8); i; i = *(_QWORD *)(i + 8) )
    {
      v9 = sub_1648700(i);
      if ( v9 != v7 )
      {
        v36.m128i_i64[0] = v9;
        sub_13E4430((__int64)&v40, &v36);
        v7 = v35;
      }
    }
    sub_164D160(v7, a2);
    if ( *(_QWORD *)(v35 + 40) )
    {
      v11 = *(unsigned __int8 *)(v35 + 16);
      v12 = (unsigned int)(v11 - 34);
      if ( (unsigned int)v12 > 0x36 || (v13 = 0x40018000000001LL, !_bittest64(&v13, v12)) )
      {
        if ( (unsigned int)(v11 - 25) > 9 && !(unsigned __int8)sub_15F3040(v35) && !(unsigned __int8)sub_15F3330(v35) )
          sub_15F20C0(v35, a2, v29, v30);
      }
    }
    if ( (_DWORD)v44 )
      goto LABEL_13;
LABEL_36:
    v15 = 0;
    goto LABEL_24;
  }
  sub_13E4430((__int64)&v40, &v35);
  if ( !(_DWORD)v44 )
    goto LABEL_36;
LABEL_13:
  v14 = 0;
  v15 = 0;
  v16 = 0;
  do
  {
    v17 = *(_QWORD *)&v43[8 * v16];
    v39 = 0;
    v36.m128i_i64[0] = v34;
    v35 = v17;
    v36.m128i_i64[1] = a3;
    v37 = a4;
    v38 = a5;
    v18 = sub_13E3350(v17, &v36, 0, 1, v10);
    if ( v18 )
    {
      v19 = v35;
      v20 = *(_QWORD *)(v35 + 8);
      if ( v20 )
      {
        do
        {
          v36.m128i_i64[0] = sub_1648700(v20);
          sub_13E4430((__int64)&v40, &v36);
          v20 = *(_QWORD *)(v20 + 8);
        }
        while ( v20 );
        v19 = v35;
      }
      v21 = v18;
      sub_164D160(v19, v18);
      v22 = v35;
      if ( !*(_QWORD *)(v35 + 40)
        || (v23 = *(unsigned __int8 *)(v35 + 16), v24 = (unsigned int)(v23 - 34), (unsigned int)v24 <= 0x36)
        && (v25 = 0x40018000000001LL, _bittest64(&v25, v24))
        || (unsigned int)(v23 - 25) <= 9
        || (unsigned __int8)sub_15F3040(v35)
        || (unsigned __int8)sub_15F3330(v22) )
      {
        v15 = 1;
      }
      else
      {
        v15 = 1;
        sub_15F20C0(v35, v21, v27, v28);
      }
    }
    v16 = (unsigned int)(v14 + 1);
    v14 = v16;
  }
  while ( (_DWORD)v16 != (_DWORD)v44 );
LABEL_24:
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  if ( (v41 & 1) == 0 )
    j___libc_free_0(v42);
  return v15;
}
