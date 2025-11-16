// Function: sub_26EC370
// Address: 0x26ec370
//
void __fastcall sub_26EC370(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r12
  __m128i v12; // rax
  float *v13; // rax
  float v14; // xmm0_4
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 i; // rbx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  float *v21; // rax
  __m128i v22; // rax
  float v23; // xmm1_4
  _QWORD *v24; // rbx
  unsigned __int64 v25; // rdi
  __int64 v26; // [rsp+18h] [rbp-C8h]
  __int64 v27; // [rsp+20h] [rbp-C0h]
  __int64 v28; // [rsp+20h] [rbp-C0h]
  __int64 v29; // [rsp+30h] [rbp-B0h]
  float v30; // [rsp+30h] [rbp-B0h]
  __int64 *v31; // [rsp+38h] [rbp-A8h]
  __m128i v32; // [rsp+40h] [rbp-A0h] BYREF
  _DWORD v33[5]; // [rsp+50h] [rbp-90h] BYREF
  char v34; // [rsp+64h] [rbp-7Ch]
  void *s; // [rsp+70h] [rbp-70h] BYREF
  __int64 v36; // [rsp+78h] [rbp-68h]
  _QWORD *v37; // [rsp+80h] [rbp-60h]
  __int64 v38; // [rsp+88h] [rbp-58h]
  int v39; // [rsp+90h] [rbp-50h]
  __int64 v40; // [rsp+98h] [rbp-48h]
  _QWORD v41[8]; // [rsp+A0h] [rbp-40h] BYREF

  v3 = sub_BC1CD0(a3, &unk_4F8D9A8, a2);
  v5 = *(_QWORD *)(a2 + 80);
  v36 = 1;
  v37 = 0;
  v31 = (__int64 *)(v3 + 8);
  s = v41;
  v38 = 0;
  v39 = 1065353216;
  v40 = 0;
  v41[0] = 0;
  v26 = a2 + 72;
  if ( v5 != a2 + 72 )
  {
    do
    {
      if ( !v5 )
        BUG();
      v6 = v5 + 24;
      v29 = v5 - 24;
      if ( *(_QWORD *)(v5 + 32) != v5 + 24 )
      {
        v27 = v5;
        v7 = *(_QWORD *)(v5 + 32);
        do
        {
          while ( 1 )
          {
            v8 = 0;
            if ( v7 )
              v8 = v7 - 24;
            sub_3143F80(v33, v8, v4);
            if ( v34 )
              break;
            v7 = *(_QWORD *)(v7 + 8);
            if ( v6 == v7 )
              goto LABEL_14;
          }
          v9 = 0;
          v10 = sub_B10CD0(v8 + 48);
          v11 = sub_26E9470(v10);
          v12.m128i_i64[0] = sub_FDD2C0(v31, v29, 0);
          v32 = v12;
          if ( v12.m128i_i8[8] )
            v9 = v32.m128i_i64[0];
          v32.m128i_i64[1] = v11;
          v32.m128i_i64[0] = v33[0];
          v13 = (float *)sub_26EC210((unsigned __int64 *)&s, &v32);
          if ( v9 < 0 )
          {
            v4 = v9 & 1;
            v14 = (float)(int)(v4 | ((unsigned __int64)v9 >> 1)) + (float)(int)(v4 | ((unsigned __int64)v9 >> 1));
          }
          else
          {
            v14 = (float)(int)v9;
          }
          *v13 = v14 + *v13;
          v7 = *(_QWORD *)(v7 + 8);
        }
        while ( v6 != v7 );
LABEL_14:
        v5 = v27;
      }
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v26 != v5 );
    if ( v26 != *(_QWORD *)(a2 + 80) )
    {
      v28 = v5;
      v15 = *(_QWORD *)(a2 + 80);
      do
      {
        if ( !v15 )
          BUG();
        v16 = *(_QWORD *)(v15 + 32);
        for ( i = v15 + 24; i != v16; v16 = *(_QWORD *)(v16 + 8) )
        {
          while ( 1 )
          {
            v18 = 0;
            if ( v16 )
              v18 = v16 - 24;
            v19 = v18;
            sub_3143F80(v33, v18, v4);
            if ( v34 )
            {
              v20 = sub_B10CD0(v19 + 48);
              v32.m128i_i64[1] = sub_26E9470(v20);
              v32.m128i_i64[0] = v33[0];
              v21 = (float *)sub_26EC210((unsigned __int64 *)&s, &v32);
              if ( *v21 != 0.0 )
                break;
            }
            v16 = *(_QWORD *)(v16 + 8);
            if ( i == v16 )
              goto LABEL_30;
          }
          v30 = *v21;
          v22.m128i_i64[0] = sub_FDD2C0(v31, v15 - 24, 0);
          v23 = 0.0;
          v32 = v22;
          if ( v22.m128i_i8[8] )
          {
            if ( v32.m128i_i64[0] < 0 )
              v23 = (float)(int)(v32.m128i_i8[0] & 1 | ((unsigned __int64)v32.m128i_i64[0] >> 1))
                  + (float)(int)(v32.m128i_i8[0] & 1 | ((unsigned __int64)v32.m128i_i64[0] >> 1));
            else
              v23 = (float)v32.m128i_i32[0];
          }
          sub_3144140(v19, v23 / v30);
        }
LABEL_30:
        v15 = *(_QWORD *)(v15 + 8);
      }
      while ( v28 != v15 );
    }
    v24 = v37;
    while ( v24 )
    {
      v25 = (unsigned __int64)v24;
      v24 = (_QWORD *)*v24;
      j_j___libc_free_0(v25);
    }
  }
  memset(s, 0, 8 * v36);
  v38 = 0;
  v37 = 0;
  if ( s != v41 )
    j_j___libc_free_0((unsigned __int64)s);
}
