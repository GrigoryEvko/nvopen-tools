// Function: sub_28532A0
// Address: 0x28532a0
//
void __fastcall sub_28532A0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  unsigned int v4; // eax
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  char *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rdi
  __int64 v18; // [rsp+10h] [rbp-A0h]
  char v19; // [rsp+28h] [rbp-88h]
  __int64 v20; // [rsp+30h] [rbp-80h]
  char *v21[2]; // [rsp+38h] [rbp-78h] BYREF
  _BYTE v22[32]; // [rsp+48h] [rbp-68h] BYREF
  __int64 v23; // [rsp+68h] [rbp-48h]
  __m128i v24; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 760);
  v4 = *(_DWORD *)(a1 + 768);
  v5 = 112LL * v4;
  v6 = v3 + v5 - 112;
  if ( a2 != (__int64 *)v6 )
  {
    v7 = *a2;
    v8 = _mm_loadu_si128((const __m128i *)(a2 + 1));
    v9 = (__int64)(a2 + 5);
    v21[0] = v22;
    v18 = v7;
    v19 = *((_BYTE *)a2 + 24);
    v20 = a2[4];
    v21[1] = (char *)0x400000000LL;
    if ( *((_DWORD *)a2 + 12) )
    {
      sub_28502F0((__int64)v21, (char **)a2 + 5, v3, v5, (__int64)v21, v9);
      v9 = (__int64)(a2 + 5);
    }
    v10 = a2[11];
    v24 = _mm_loadu_si128((const __m128i *)a2 + 6);
    v23 = v10;
    *a2 = *(_QWORD *)v6;
    a2[1] = *(_QWORD *)(v6 + 8);
    *((_BYTE *)a2 + 16) = *(_BYTE *)(v6 + 16);
    *((_BYTE *)a2 + 24) = *(_BYTE *)(v6 + 24);
    a2[4] = *(_QWORD *)(v6 + 32);
    sub_28502F0(v9, (char **)(v6 + 40), v3, v5, (__int64)v21, v9);
    a2[11] = *(_QWORD *)(v6 + 88);
    a2[12] = *(_QWORD *)(v6 + 96);
    *((_BYTE *)a2 + 104) = *(_BYTE *)(v6 + 104);
    *(_QWORD *)v6 = v18;
    *(_QWORD *)(v6 + 8) = v8.m128i_i64[0];
    *(_BYTE *)(v6 + 16) = v8.m128i_i8[8];
    *(_BYTE *)(v6 + 24) = v19;
    *(_QWORD *)(v6 + 32) = v20;
    sub_28502F0(v6 + 40, v21, v11, v12, (__int64)v21, v13);
    v14 = v21[0];
    *(_QWORD *)(v6 + 88) = v23;
    *(_QWORD *)(v6 + 96) = v24.m128i_i64[0];
    *(_BYTE *)(v6 + 104) = v24.m128i_i8[8];
    if ( v14 != v22 )
      _libc_free((unsigned __int64)v14);
    v4 = *(_DWORD *)(a1 + 768);
    v3 = *(_QWORD *)(a1 + 760);
  }
  v15 = v4 - 1;
  *(_DWORD *)(a1 + 768) = v15;
  v16 = 112 * v15 + v3;
  v17 = *(_QWORD *)(v16 + 40);
  if ( v17 != v16 + 56 )
    _libc_free(v17);
}
