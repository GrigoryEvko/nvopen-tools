// Function: sub_28644C0
// Address: 0x28644c0
//
void __fastcall sub_28644C0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v10; // rax
  __int64 *v11; // rsi
  __int64 v12; // r13
  bool v13; // al
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i v18; // xmm0
  char v19; // al
  __m128i v20; // xmm1
  __int64 v21; // rcx
  char v22; // dl
  __int64 v23; // rsi
  __int64 *v24; // rdi
  unsigned int v25; // eax
  unsigned int v26; // r9d
  char v27; // r8
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v32; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v33; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v34; // [rsp+38h] [rbp-98h]
  char v35; // [rsp+48h] [rbp-88h]
  __int64 v36; // [rsp+50h] [rbp-80h]
  unsigned __int64 v37[2]; // [rsp+58h] [rbp-78h] BYREF
  _BYTE v38[32]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v39; // [rsp+88h] [rbp-48h]
  __m128i v40; // [rsp+90h] [rbp-40h]

  if ( a6 )
    v10 = *(_QWORD *)(a4 + 88);
  else
    v10 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 8 * a5);
  v11 = *(__int64 **)(a1 + 8);
  v32 = v10;
  v12 = sub_28569D0(&v32, v11);
  v13 = sub_D968A0(v32);
  if ( v12 && !v13 )
  {
    v18 = _mm_loadu_si128((const __m128i *)(a4 + 8));
    v33 = *(_QWORD *)a4;
    v19 = *(_BYTE *)(a4 + 24);
    v34 = v18;
    v35 = v19;
    v36 = *(_QWORD *)(a4 + 32);
    v37[0] = (unsigned __int64)v38;
    v37[1] = 0x400000000LL;
    if ( *(_DWORD *)(a4 + 48) )
      sub_2850210((__int64)v37, a4 + 40, v14, v15, v16, v17);
    v20 = _mm_loadu_si128((const __m128i *)(a4 + 96));
    v21 = *(_QWORD *)(a2 + 728);
    v22 = *(_BYTE *)(a2 + 720);
    v23 = *(_QWORD *)(a2 + 712);
    v24 = *(__int64 **)(a1 + 48);
    v39 = *(_QWORD *)(a4 + 88);
    v25 = *(_DWORD *)(a2 + 48);
    v26 = *(_DWORD *)(a2 + 32);
    v40 = v20;
    v27 = *(_BYTE *)(a2 + 736);
    v33 = v12;
    if ( sub_2850770(v24, v23, v22, v21, v27, v26, *(_QWORD *)(a2 + 40), v25, (__int64)&v33) )
    {
      if ( a6 )
        v39 = v32;
      else
        *(_QWORD *)(v37[0] + 8 * a5) = v32;
      sub_2862B30(a1, a2, a3, (unsigned __int64)&v33, v28, v29);
    }
    if ( (_BYTE *)v37[0] != v38 )
      _libc_free(v37[0]);
  }
}
