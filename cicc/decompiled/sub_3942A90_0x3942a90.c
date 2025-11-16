// Function: sub_3942A90
// Address: 0x3942a90
//
__int64 __fastcall sub_3942A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r12
  __int64 result; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r14
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 **v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // rbx
  size_t v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // [rsp+10h] [rbp-F0h] BYREF
  char v22; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v23; // [rsp+30h] [rbp-D0h] BYREF
  size_t v24; // [rsp+38h] [rbp-C8h]
  char v25; // [rsp+40h] [rbp-C0h]
  __m128i v26; // [rsp+50h] [rbp-B0h] BYREF
  __int128 v27; // [rsp+60h] [rbp-A0h]
  int v28; // [rsp+78h] [rbp-88h] BYREF
  unsigned __int64 v29; // [rsp+80h] [rbp-80h]
  int *v30; // [rsp+88h] [rbp-78h]
  int *v31; // [rsp+90h] [rbp-70h]
  __int64 v32; // [rsp+98h] [rbp-68h]
  int v33; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v34; // [rsp+B0h] [rbp-50h]
  int *v35; // [rsp+B8h] [rbp-48h]
  int *v36; // [rsp+C0h] [rbp-40h]
  __int64 v37; // [rsp+C8h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 72) >= *(_QWORD *)(a1 + 80) )
  {
LABEL_15:
    sub_393D180(a1, a2, a3, a4, a5, a6);
    return 0;
  }
  else
  {
    v6 = (_QWORD *)a1;
    while ( 1 )
    {
      sub_393FF90((__int64)&v21, v6);
      if ( (v22 & 1) != 0 )
      {
        result = (unsigned int)v21;
        if ( (_DWORD)v21 )
          break;
      }
      (*(void (__fastcall **)(unsigned __int8 **, _QWORD *))(*v6 + 48LL))(&v23, v6);
      if ( (v25 & 1) != 0 )
      {
        result = (unsigned int)v23;
        if ( (_DWORD)v23 )
          break;
      }
      v26 = 0;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v30 = &v28;
      v31 = &v28;
      v32 = 0;
      v33 = 0;
      v34 = 0;
      v35 = &v33;
      v36 = &v33;
      v37 = 0;
      v8 = *(_QWORD *)sub_3940400((__int64)(v6 + 1), v23, v24);
      *(__m128i *)(v8 + 8) = _mm_loadu_si128(&v26);
      v9 = *(_QWORD *)(v8 + 56);
      *(_OWORD *)(v8 + 24) = v27;
      sub_393DB20(v9);
      *(_QWORD *)(v8 + 56) = 0;
      *(_QWORD *)(v8 + 64) = v8 + 48;
      *(_QWORD *)(v8 + 72) = v8 + 48;
      *(_QWORD *)(v8 + 80) = 0;
      if ( v29 )
      {
        *(_DWORD *)(v8 + 48) = v28;
        v10 = v29;
        *(_QWORD *)(v8 + 56) = v29;
        *(_QWORD *)(v8 + 64) = v30;
        *(_QWORD *)(v8 + 72) = v31;
        *(_QWORD *)(v10 + 8) = v8 + 48;
        *(_QWORD *)(v8 + 80) = v32;
        v29 = 0;
        v30 = &v28;
        v31 = &v28;
        v32 = 0;
      }
      v11 = *(_QWORD *)(v8 + 104);
      while ( v11 )
      {
        v12 = v11;
        sub_393DEF0(*(_QWORD **)(v11 + 24));
        v13 = *(_QWORD **)(v11 + 56);
        v11 = *(_QWORD *)(v11 + 16);
        sub_393E140(v13);
        j_j___libc_free_0(v12);
      }
      *(_QWORD *)(v8 + 104) = 0;
      *(_QWORD *)(v8 + 112) = v8 + 96;
      *(_QWORD *)(v8 + 120) = v8 + 96;
      *(_QWORD *)(v8 + 128) = 0;
      if ( v34 )
      {
        *(_DWORD *)(v8 + 96) = v33;
        v14 = v34;
        *(_QWORD *)(v8 + 104) = v34;
        *(_QWORD *)(v8 + 112) = v35;
        *(_QWORD *)(v8 + 120) = v36;
        *(_QWORD *)(v14 + 8) = v8 + 96;
        *(_QWORD *)(v8 + 128) = v37;
      }
      sub_393DB20(v29);
      v15 = (unsigned __int64 **)sub_3940400((__int64)(v6 + 1), v23, v24);
      v16 = v21;
      v17 = *v15;
      v18 = v24;
      v17[1] = (unsigned __int64)v23;
      v19 = v17[4];
      v17[2] = v18;
      v20 = sub_393FEE0(v16, 1u, v19, (bool *)v26.m128i_i8);
      a2 = (__int64)(v17 + 1);
      a1 = (__int64)v6;
      v17[4] = v20;
      result = sub_39422E0(v6, v17 + 1);
      if ( (_DWORD)result )
        break;
      if ( v6[10] <= v6[9] )
        goto LABEL_15;
    }
  }
  return result;
}
