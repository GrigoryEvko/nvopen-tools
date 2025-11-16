// Function: sub_1420A50
// Address: 0x1420a50
//
unsigned __int64 __fastcall sub_1420A50(__int64 a1, __m128i *a2, __int64 a3, unsigned __int32 a4)
{
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  bool v13; // al
  unsigned __int64 result; // rax
  int v15; // r15d
  __int64 v16; // rax
  unsigned int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // [rsp+0h] [rbp-130h] BYREF
  __m128i v21; // [rsp+8h] [rbp-128h] BYREF
  __m128i v22; // [rsp+18h] [rbp-118h] BYREF
  __int64 v23; // [rsp+28h] [rbp-108h]
  __m128i *v24; // [rsp+30h] [rbp-100h]
  int v25; // [rsp+38h] [rbp-F8h]
  __int64 v26; // [rsp+40h] [rbp-F0h]
  __int64 v27; // [rsp+48h] [rbp-E8h]
  __int64 v28; // [rsp+50h] [rbp-E0h]
  __int64 v29; // [rsp+58h] [rbp-D8h]
  __int64 v30; // [rsp+60h] [rbp-D0h]
  __m128i *v31; // [rsp+68h] [rbp-C8h]
  bool i; // [rsp+70h] [rbp-C0h]
  __int64 v33; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v34; // [rsp+88h] [rbp-A8h]
  __m128i v35; // [rsp+98h] [rbp-98h]
  __int64 v36; // [rsp+A8h] [rbp-88h]
  __m128i *v37; // [rsp+B0h] [rbp-80h]
  int v38; // [rsp+B8h] [rbp-78h]
  __int64 v39; // [rsp+C0h] [rbp-70h]
  __int64 v40; // [rsp+C8h] [rbp-68h]
  __int64 v41; // [rsp+D0h] [rbp-60h]
  __int64 v42; // [rsp+D8h] [rbp-58h]
  __int64 v43; // [rsp+E0h] [rbp-50h]
  __m128i *v44; // [rsp+E8h] [rbp-48h]
  bool v45; // [rsp+F0h] [rbp-40h]

  v7 = (__int64 *)(*(_QWORD *)(a1 + 32) + ((unsigned __int64)a4 << 6));
  v8 = *v7;
  v9 = v7[1];
  v33 = 0;
  v10 = v7[2];
  v11 = v7[3];
  v34.m128i_i64[0] = 0;
  v34.m128i_i64[1] = -1;
  v12 = v7[4];
  v35 = 0u;
  v43 = v12;
  v13 = 0;
  v36 = 0;
  v37 = a2;
  v38 = 0;
  v39 = v8;
  v40 = v9;
  v41 = v10;
  v42 = v11;
  v44 = a2;
  if ( a2 )
    v13 = a2[1].m128i_i8[0] == 23;
  v45 = v13;
  sub_1420880((__int64)&v33);
  result = (unsigned __int64)v37;
  v20 = v33;
  v24 = v37;
  v21 = v34;
  v22 = v35;
  v23 = v36;
  v25 = v38;
  v26 = v39;
  v27 = v40;
  v28 = v41;
  v29 = v42;
  v30 = v43;
  v31 = v44;
  for ( i = v45; v24; result = (unsigned __int64)v24 )
  {
    v15 = *(_DWORD *)(a1 + 40);
    v16 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v16 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, a3 + 16, 0, 4);
      v16 = *(unsigned int *)(a3 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a3 + 4 * v16) = v15;
    ++*(_DWORD *)(a3 + 8);
    v17 = *(_DWORD *)(a1 + 40);
    if ( v17 >= *(_DWORD *)(a1 + 44) )
    {
      sub_1420280(a1 + 32);
      v17 = *(_DWORD *)(a1 + 40);
    }
    result = *(_QWORD *)(a1 + 32) + ((unsigned __int64)v17 << 6);
    if ( result )
    {
      v18 = v20;
      *(__m128i *)result = _mm_loadu_si128(&v21);
      *(__m128i *)(result + 16) = _mm_loadu_si128(&v22);
      v19 = v23;
      *(_QWORD *)(result + 40) = v18;
      *(_QWORD *)(result + 32) = v19;
      *(_QWORD *)(result + 48) = v18;
      *(_BYTE *)(result + 60) = 1;
      *(_DWORD *)(result + 56) = a4;
      v17 = *(_DWORD *)(a1 + 40);
    }
    *(_DWORD *)(a1 + 40) = v17 + 1;
    if ( v24[1].m128i_i8[0] != 23 )
      break;
    result = (unsigned int)(v25 + 1);
    v25 = result;
    if ( (unsigned int)result >= (v24[1].m128i_i32[1] & 0xFFFFFFFu) )
      break;
    sub_1420880((__int64)&v20);
  }
  return result;
}
