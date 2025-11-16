// Function: sub_33FE730
// Address: 0x33fe730
//
__int64 __fastcall sub_33FE730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __m128i a6)
{
  unsigned __int16 v7; // bx
  __int64 v8; // rdx
  __int64 v9; // r12
  __m128i v10; // xmm0
  _DWORD *v12; // rax
  int v13; // ebx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *i; // rbx
  _DWORD *v18; // [rsp+0h] [rbp-A0h]
  _DWORD *v19; // [rsp+0h] [rbp-A0h]
  _DWORD *v20; // [rsp+0h] [rbp-A0h]
  double v21; // [rsp+8h] [rbp-98h]
  __int64 v22; // [rsp+10h] [rbp-90h] BYREF
  __int64 v23; // [rsp+18h] [rbp-88h]
  unsigned __int16 v24; // [rsp+20h] [rbp-80h] BYREF
  __int64 v25; // [rsp+28h] [rbp-78h]
  void *v26; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v27; // [rsp+38h] [rbp-68h]
  __int64 v28[10]; // [rsp+50h] [rbp-50h] BYREF

  v7 = a3;
  v22 = a3;
  v23 = a4;
  v21 = *(double *)a6.m128i_i64;
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 17) > 0xD3u )
    {
LABEL_3:
      v8 = v23;
      goto LABEL_4;
    }
    v13 = (unsigned __int16)a3;
    v8 = 0;
    v7 = word_4456580[v13 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v22) )
      goto LABEL_3;
    v7 = sub_3009970((__int64)&v22, a2, v14, v15, v16);
  }
LABEL_4:
  v24 = v7;
  v25 = v8;
  if ( v7 == 12 )
  {
    v10 = 0;
    v19 = sub_C33310();
    *(float *)v10.m128i_i32 = v21;
    sub_C3B170((__int64)v28, v10);
    sub_C407B0(&v26, v28, v19);
    sub_C338F0((__int64)v28);
    v9 = sub_33FE6E0(a1, (__int64 *)&v26, a2, v22, v23, a5, v10);
    if ( v26 == sub_C33340() )
    {
      if ( v27 )
      {
        for ( i = &v27[3 * *(v27 - 1)]; v27 != i; sub_91D830(i) )
          i -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v26);
    }
  }
  else
  {
    if ( v7 == 13 )
    {
      a6 = (__m128i)a6.m128i_u64[0];
      v18 = sub_C33320();
      sub_C3B1B0((__int64)v28, *(double *)a6.m128i_i64);
      sub_C407B0(&v26, v28, v18);
      sub_C338F0((__int64)v28);
    }
    else
    {
      if ( (unsigned __int16)(v7 - 10) > 1u && (unsigned __int16)(v7 - 14) > 2u )
        BUG();
      a6 = (__m128i)a6.m128i_u64[0];
      v20 = sub_C33320();
      sub_C3B1B0((__int64)v28, *(double *)a6.m128i_i64);
      sub_C407B0(&v26, v28, v20);
      sub_C338F0((__int64)v28);
      v12 = sub_300AC80(&v24, (__int64)v28);
      sub_C41640((__int64 *)&v26, v12, 1, (bool *)v28);
    }
    v9 = sub_33FE6E0(a1, (__int64 *)&v26, a2, v22, v23, a5, a6);
    sub_91D830(&v26);
  }
  return v9;
}
