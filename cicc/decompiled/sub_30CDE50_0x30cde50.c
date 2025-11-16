// Function: sub_30CDE50
// Address: 0x30cde50
//
void __fastcall sub_30CDE50(__int64 *a1, __int64 a2, char **a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r12
  char *v9; // r14
  size_t v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r12
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 *v21; // r14
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  unsigned __int64 *v24; // r13
  unsigned __int64 *v25; // r12
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-498h]
  __m128i v30; // [rsp+10h] [rbp-490h] BYREF
  unsigned __int64 v31[2]; // [rsp+20h] [rbp-480h] BYREF
  __int64 v32; // [rsp+30h] [rbp-470h] BYREF
  __int64 *v33; // [rsp+40h] [rbp-460h]
  __int64 v34; // [rsp+50h] [rbp-450h] BYREF
  unsigned __int64 v35[2]; // [rsp+70h] [rbp-430h] BYREF
  __int64 v36; // [rsp+80h] [rbp-420h] BYREF
  __int64 *v37; // [rsp+90h] [rbp-410h]
  __int64 v38; // [rsp+A0h] [rbp-400h] BYREF
  unsigned __int64 v39[2]; // [rsp+C0h] [rbp-3E0h] BYREF
  __int64 v40; // [rsp+D0h] [rbp-3D0h] BYREF
  __int64 *v41; // [rsp+E0h] [rbp-3C0h]
  __int64 v42; // [rsp+F0h] [rbp-3B0h] BYREF
  void *v43; // [rsp+110h] [rbp-390h] BYREF
  int v44; // [rsp+118h] [rbp-388h]
  char v45; // [rsp+11Ch] [rbp-384h]
  __int64 v46; // [rsp+120h] [rbp-380h]
  __m128i v47; // [rsp+128h] [rbp-378h]
  __int64 v48; // [rsp+138h] [rbp-368h]
  __m128i v49; // [rsp+140h] [rbp-360h]
  __m128i v50; // [rsp+150h] [rbp-350h]
  unsigned __int64 *v51; // [rsp+160h] [rbp-340h] BYREF
  __int64 v52; // [rsp+168h] [rbp-338h]
  _BYTE v53[324]; // [rsp+170h] [rbp-330h] BYREF
  int v54; // [rsp+2B4h] [rbp-1ECh]
  __int64 v55; // [rsp+2B8h] [rbp-1E8h]
  _QWORD v56[10]; // [rsp+2C0h] [rbp-1E0h] BYREF
  unsigned __int64 *v57; // [rsp+310h] [rbp-190h]
  unsigned int v58; // [rsp+318h] [rbp-188h]
  char v59; // [rsp+320h] [rbp-180h] BYREF

  v5 = *a1;
  v6 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v6)
    || (v27 = sub_B2BE50(v5),
        v28 = sub_B6F970(v27),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v28 + 48LL))(v28)) )
  {
    v29 = *(_QWORD *)(a2 + 40);
    sub_B157E0((__int64)&v30, (_QWORD *)(a2 + 32));
    sub_B17640((__int64)v56, *(_QWORD *)(*(_QWORD *)(a2 + 8) + 40LL), (__int64)"NotInlined", 10, &v30, v29);
    sub_B18290((__int64)v56, "'", 1u);
    sub_B16080((__int64)v39, "Callee", 6, *(unsigned __int8 **)(a2 + 24));
    v7 = sub_2445430((__int64)v56, (__int64)v39);
    sub_B18290(v7, "' is not AlwaysInline into '", 0x1Cu);
    sub_B16080((__int64)v35, "Caller", 6, *(unsigned __int8 **)(a2 + 16));
    v8 = sub_2445430(v7, (__int64)v35);
    sub_B18290(v8, "': ", 3u);
    v9 = *a3;
    v10 = 0;
    if ( v9 )
      v10 = strlen(v9);
    sub_B16430((__int64)v31, "Reason", 6u, v9, v10);
    v15 = sub_2445430(v8, (__int64)v31);
    v16 = _mm_loadu_si128((const __m128i *)(v15 + 24));
    v17 = _mm_loadu_si128((const __m128i *)(v15 + 48));
    v44 = *(_DWORD *)(v15 + 8);
    v18 = _mm_loadu_si128((const __m128i *)(v15 + 64));
    v45 = *(_BYTE *)(v15 + 12);
    v19 = *(_QWORD *)(v15 + 16);
    v47 = v16;
    v46 = v19;
    v43 = &unk_49D9D40;
    v20 = *(_QWORD *)(v15 + 40);
    v51 = (unsigned __int64 *)v53;
    v48 = v20;
    v52 = 0x400000000LL;
    LODWORD(v20) = *(_DWORD *)(v15 + 88);
    v49 = v17;
    v50 = v18;
    if ( (_DWORD)v20 )
      sub_30CDBD0((__int64)&v51, v15 + 80, v11, v12, v13, v14);
    v53[320] = *(_BYTE *)(v15 + 416);
    v54 = *(_DWORD *)(v15 + 420);
    v55 = *(_QWORD *)(v15 + 424);
    v43 = &unk_49D9DB0;
    if ( v33 != &v34 )
      j_j___libc_free_0((unsigned __int64)v33);
    if ( (__int64 *)v31[0] != &v32 )
      j_j___libc_free_0(v31[0]);
    if ( v37 != &v38 )
      j_j___libc_free_0((unsigned __int64)v37);
    if ( (__int64 *)v35[0] != &v36 )
      j_j___libc_free_0(v35[0]);
    if ( v41 != &v42 )
      j_j___libc_free_0((unsigned __int64)v41);
    if ( (__int64 *)v39[0] != &v40 )
      j_j___libc_free_0(v39[0]);
    v21 = v57;
    v56[0] = &unk_49D9D40;
    v22 = &v57[10 * v58];
    if ( v57 != v22 )
    {
      do
      {
        v22 -= 10;
        v23 = v22[4];
        if ( (unsigned __int64 *)v23 != v22 + 6 )
          j_j___libc_free_0(v23);
        if ( (unsigned __int64 *)*v22 != v22 + 2 )
          j_j___libc_free_0(*v22);
      }
      while ( v21 != v22 );
      v22 = v57;
    }
    if ( v22 != (unsigned __int64 *)&v59 )
      _libc_free((unsigned __int64)v22);
    sub_1049740(a1, (__int64)&v43);
    v24 = v51;
    v43 = &unk_49D9D40;
    v25 = &v51[10 * (unsigned int)v52];
    if ( v51 != v25 )
    {
      do
      {
        v25 -= 10;
        v26 = v25[4];
        if ( (unsigned __int64 *)v26 != v25 + 6 )
          j_j___libc_free_0(v26);
        if ( (unsigned __int64 *)*v25 != v25 + 2 )
          j_j___libc_free_0(*v25);
      }
      while ( v24 != v25 );
      v25 = v51;
    }
    if ( v25 != (unsigned __int64 *)v53 )
      _libc_free((unsigned __int64)v25);
  }
}
