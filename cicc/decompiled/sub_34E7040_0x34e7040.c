// Function: sub_34E7040
// Address: 0x34e7040
//
void __fastcall sub_34E7040(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned int v5; // ebx
  char *v6; // rax
  unsigned __int16 *v7; // r15
  __int64 v8; // r8
  __int64 i; // r10
  __int64 v10; // rbx
  char *v11; // r9
  unsigned int v12; // eax
  _WORD *v13; // rdx
  __int64 v14; // rax
  unsigned __int16 *v15; // r14
  __int32 v16; // r12d
  __int64 v17; // r15
  __int16 v18; // bx
  __int64 v19; // r9
  __int64 v20; // rax
  __int16 *v21; // rdi
  __int32 j; // r8d
  unsigned int v23; // eax
  _WORD *v24; // rdx
  int v25; // eax
  unsigned int v26; // eax
  _WORD *v27; // rdx
  unsigned __int16 *v28; // [rsp+20h] [rbp-100h]
  __int64 v29; // [rsp+20h] [rbp-100h]
  _BYTE *v30; // [rsp+28h] [rbp-F8h]
  __int64 v31; // [rsp+28h] [rbp-F8h]
  __m128i v32; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v33; // [rsp+40h] [rbp-E0h]
  __int64 v34; // [rsp+48h] [rbp-D8h]
  __int64 v35; // [rsp+50h] [rbp-D0h]
  _BYTE *v36; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v38; // [rsp+70h] [rbp-B0h]
  _BYTE v39[16]; // [rsp+78h] [rbp-A8h] BYREF
  char *v40; // [rsp+88h] [rbp-98h]
  unsigned int v41; // [rsp+90h] [rbp-90h]
  unsigned __int16 *v42; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v43; // [rsp+A8h] [rbp-78h]
  _BYTE v44[112]; // [rsp+B0h] [rbp-70h] BYREF

  v2 = sub_2E88D60(a1);
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 16) + 200LL))(*(_QWORD *)(v2 + 16));
  v38 = 8;
  v40 = 0;
  v4 = v3;
  v41 = 0;
  v5 = *(_DWORD *)(v3 + 16);
  v36 = v39;
  v37 = 0;
  if ( v5 )
  {
    v6 = (char *)_libc_calloc(v5, 1u);
    if ( !v6 )
      sub_C64F00("Allocation failed", 1u);
    v40 = v6;
    v41 = v5;
  }
  v7 = *(unsigned __int16 **)(a2 + 8);
  v8 = (__int64)&v7[*(_QWORD *)(a2 + 16)];
  if ( v7 != (unsigned __int16 *)v8 )
  {
    for ( i = 0; ; i = v37 )
    {
      v10 = *v7;
      v11 = &v40[v10];
      v12 = (unsigned __int8)v40[v10];
      if ( v12 >= (unsigned int)i )
        goto LABEL_13;
      while ( 1 )
      {
        v13 = &v36[2 * v12];
        if ( (_WORD)v10 == *v13 )
          break;
        v12 += 256;
        if ( (unsigned int)i <= v12 )
          goto LABEL_13;
      }
      if ( v13 == (_WORD *)&v36[2 * i] )
      {
LABEL_13:
        *v11 = i;
        v14 = v37;
        if ( v37 + 1 > v38 )
        {
          v29 = v8;
          sub_C8D290((__int64)&v36, v39, v37 + 1, 2u, v8, (__int64)v11);
          v14 = v37;
          v8 = v29;
        }
        ++v7;
        *(_WORD *)&v36[2 * v14] = v10;
        ++v37;
        if ( (unsigned __int16 *)v8 == v7 )
          break;
      }
      else if ( (unsigned __int16 *)v8 == ++v7 )
      {
        break;
      }
    }
  }
  v42 = (unsigned __int16 *)v44;
  v43 = 0x400000000LL;
  sub_3507DB0(a2, a1, &v42);
  v15 = v42;
  v28 = &v42[8 * (unsigned int)v43];
  if ( v28 != v42 )
  {
    do
    {
      v16 = *v15;
      v17 = *(_QWORD *)(*((_QWORD *)v15 + 1) + 16LL);
      v30 = (_BYTE *)*((_QWORD *)v15 + 1);
      v18 = *v15;
      v19 = sub_2E88D60(v17);
      v20 = (unsigned __int16)v16;
      if ( *v30 == 12 )
      {
        v26 = (unsigned __int8)v40[(unsigned __int16)v16];
        if ( v26 < (unsigned int)v37 )
        {
          while ( 1 )
          {
            v27 = &v36[2 * v26];
            if ( v18 == *v27 )
              break;
            v26 += 256;
            if ( (unsigned int)v37 <= v26 )
              goto LABEL_44;
          }
          if ( v27 != (_WORD *)&v36[2 * v37] )
          {
            v32.m128i_i64[0] = 0x20000000;
            v31 = v19;
            v33 = 0;
            v32.m128i_i32[2] = v16;
            v34 = 0;
            v35 = 0;
            sub_2E8EAD0(v17, v19, &v32);
            v19 = v31;
          }
        }
LABEL_44:
        v32.m128i_i64[0] = 805306368;
        v33 = 0;
        v32.m128i_i32[2] = v16;
        v34 = 0;
        v35 = 0;
        sub_2E8EAD0(v17, v19, &v32);
      }
      else
      {
        v21 = (__int16 *)(*(_QWORD *)(v4 + 56)
                        + 2LL * *(unsigned int *)(*(_QWORD *)(v4 + 8) + 24LL * (unsigned __int16)v16 + 4));
        if ( v21 )
        {
          for ( j = v16; ; v18 = j )
          {
            v23 = (unsigned __int8)v40[v20];
            if ( v23 < (unsigned int)v37 )
            {
              while ( 1 )
              {
                v24 = &v36[2 * v23];
                if ( *v24 == v18 )
                  break;
                v23 += 256;
                if ( (unsigned int)v37 <= v23 )
                  goto LABEL_36;
              }
              if ( &v36[2 * v37] != (_BYTE *)v24 )
                break;
            }
LABEL_36:
            v25 = *v21++;
            if ( !(_WORD)v25 )
              goto LABEL_27;
            j += v25;
            v20 = (unsigned __int16)j;
          }
          if ( v21 )
          {
            v32.m128i_i64[0] = 0x20000000;
            v33 = 0;
            v32.m128i_i32[2] = v16;
            v34 = 0;
            v35 = 0;
            sub_2E8EAD0(v17, v19, &v32);
          }
        }
      }
LABEL_27:
      v15 += 8;
    }
    while ( v28 != v15 );
    v15 = v42;
  }
  if ( v15 != (unsigned __int16 *)v44 )
    _libc_free((unsigned __int64)v15);
  if ( v40 )
    _libc_free((unsigned __int64)v40);
  if ( v36 != v39 )
    _libc_free((unsigned __int64)v36);
}
