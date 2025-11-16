// Function: sub_806F60
// Address: 0x806f60
//
__int64 sub_806F60()
{
  _QWORD *v0; // rax
  __int64 v1; // r12
  __int64 v2; // rax
  __m128i *v3; // rax
  __int64 v4; // rbx
  __m128i *v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rax
  __m128i *v8; // r13
  __int64 v9; // rax
  _QWORD *v10; // rax
  int *v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  __m128i *v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // r15
  _QWORD *v18; // rax
  _QWORD *v19; // r14
  const __m128i **v20; // rax
  __int64 v21; // r13
  char v22; // al
  char *v23; // r14
  __int64 *v24; // r13
  __m128i *v25; // r15
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v29; // rax
  int v30; // edi
  __int64 i; // rax
  __int64 **v32; // rbx
  __m128i *v33; // r14
  _QWORD *v34; // rax
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // rbx
  _QWORD *v39; // rax
  _QWORD *v40; // rdi
  _QWORD *v41; // rax
  __m128i *v42; // r14
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  char v45; // al
  _QWORD *v46; // rbx
  __int64 j; // rax
  unsigned __int64 v48; // rsi
  _QWORD *v49; // rbx
  _QWORD *v50; // rax
  __int64 *v51; // rsi
  char *v52; // rdi
  _QWORD *v53; // rax
  int v54; // r14d
  __int64 v55; // [rsp+8h] [rbp-168h]
  __m128i *v56; // [rsp+10h] [rbp-160h]
  __int64 *v57; // [rsp+18h] [rbp-158h]
  __int64 v58; // [rsp+20h] [rbp-150h]
  __int64 v59; // [rsp+30h] [rbp-140h]
  __int64 v60; // [rsp+38h] [rbp-138h]
  _QWORD *v61; // [rsp+40h] [rbp-130h]
  _QWORD *v62; // [rsp+40h] [rbp-130h]
  _QWORD *v63; // [rsp+40h] [rbp-130h]
  __int64 v64; // [rsp+40h] [rbp-130h]
  unsigned int v65; // [rsp+54h] [rbp-11Ch] BYREF
  const __m128i *v66; // [rsp+58h] [rbp-118h] BYREF
  int v67[8]; // [rsp+60h] [rbp-110h] BYREF
  _BYTE v68[240]; // [rsp+80h] [rbp-F0h] BYREF

  v55 = qword_4F07288;
  v0 = (_QWORD *)sub_7E1C10();
  v1 = sub_72D2E0(v0);
  v2 = sub_72CBE0();
  v56 = sub_7F7840("__nv_cudaEntityRegisterCallback", 2, v2, v1);
  sub_7362F0((__int64)v56, 0);
  sub_7604D0((__int64)v56, 0xBu);
  sub_7605A0((__int64)v56);
  v57 = sub_7F54F0((__int64)v56, 0, 0, &v65);
  v3 = sub_7E2270(v1);
  v57[5] = (__int64)v3;
  v4 = (__int64)v3;
  *(_BYTE *)(v57[4] + 197) |= 0x60u;
  sub_7E1740(v57[10], (__int64)v67);
  sub_7F6C60((__int64)v57, v65, (__int64)v68);
  v5 = (__m128i *)sub_73E830(v4);
  v6 = sub_72CBE0();
  v7 = sub_7F8900("__nv_dummy_param_ref", (__m128i **)&qword_4F18A40, v6, v5);
  sub_7E69E0(v7, v67);
  v8 = (__m128i *)sub_73E830(v4);
  v9 = sub_72CBE0();
  v10 = sub_7F8900("__nv_save_fatbinhandle_for_managed_rt", (__m128i **)&qword_4F18A38, v9, v8);
  v11 = v67;
  sub_7E69E0(v10, v67);
  for ( ; qword_4F18A30; qword_4F18A30 = *(_QWORD *)(qword_4F18A30 + 120) )
  {
    v14 = (__m128i *)sub_73E830(v4);
    v15 = sub_731330(*(_QWORD *)(qword_4F18A30 + 128));
    v16 = qword_4F18A30;
    v14[1].m128i_i64[0] = (__int64)v15;
    v17 = v15;
    v18 = sub_731330(v16);
    v17[2] = v18;
    v19 = v18;
    v20 = *(const __m128i ***)(qword_4F18A30 + 328);
    if ( v20 && *v20 )
      v19[2] = sub_73A720(*v20, (__int64)v67);
    else
      v19[2] = sub_73A830(-1, 5u);
    v12 = sub_72BA30(5u);
    v13 = sub_7F8900("__cudaRegisterEntry", (__m128i **)&qword_4F18A48, (__int64)v12, v14);
    v11 = v67;
    sub_7E69E0(v13, v67);
  }
  v21 = *(_QWORD *)(qword_4F07288 + 112);
  if ( v21 )
  {
    v58 = v4;
    do
    {
      v22 = *(_BYTE *)(v21 + 156);
      if ( (v22 & 3) != 1
        || (*(_BYTE *)(v21 + 170) & 0x10) != 0 && (*(_BYTE *)(v21 + 170) & 0x60) != 0
        || (v22 & 0x20) != 0
        || *(char *)(v21 + 173) < 0
        || ((v11 = (int *)HIDWORD(qword_4D045BC), HIDWORD(qword_4D045BC)) || (v22 & 0x10) == 0)
        && (*(_BYTE *)(v21 + 136) == 1 || !*(_QWORD *)v21 || (*(_BYTE *)(*(_QWORD *)v21 + 81LL) & 2) == 0) )
      {
        if ( (unsigned int)sub_8D2FF0(*(_QWORD *)(v21 + 120), v11)
          && (!HIDWORD(qword_4D045BC) || *(_BYTE *)(v21 + 136) != 1) )
        {
          for ( i = *(_QWORD *)(v21 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          v32 = *(__int64 ***)(*(_QWORD *)(i + 168) + 168LL);
          v33 = (__m128i *)sub_73E830(v58);
          v59 = **v32;
          v60 = sub_620FD0((*v32)[4], &v66);
          v34 = sub_73E830(v21);
          v33[1].m128i_i64[0] = (__int64)v34;
          v61 = v34;
          v35 = sub_73A830(v60, 5u);
          v36 = v61;
          v62 = v35;
          v36[2] = v35;
          if ( v59 )
          {
            v37 = sub_620FD0(*(_QWORD *)(**v32 + 32), &v66);
            v38 = sub_73A830(v37, 5u);
            v62[2] = v38;
            v38[2] = sub_73A830(*(_BYTE *)(v21 + 136) == 1, 5u);
            v39 = sub_72BA30(5u);
            v40 = sub_7F8900("__cudaRegisterGlobalTexture", (__m128i **)&qword_4F18A60, (__int64)v39, v33);
          }
          else
          {
            v35[2] = sub_73A830(*(_BYTE *)(v21 + 136) == 1, 5u);
            v41 = sub_72BA30(5u);
            v40 = sub_7F8900("__cudaRegisterGlobalSurface", (__m128i **)&qword_4F18A58, (__int64)v41, v33);
          }
          v11 = v67;
          sub_7E69E0(v40, v67);
        }
      }
      else
      {
        v42 = (__m128i *)sub_73E830(v58);
        v66 = (const __m128i *)sub_724DC0();
        v43 = sub_73E830(v21);
        v42[1].m128i_i64[0] = (__int64)v43;
        v44 = v43;
        v45 = *(_BYTE *)(v21 + 136);
        if ( !HIDWORD(qword_4D045BC) && v45 == 1 )
        {
          if ( (unsigned int)sub_8D23E0(*(_QWORD *)(v21 + 120)) )
            *(_QWORD *)v42[1].m128i_i64[0] = *(_QWORD *)(v21 + 120);
          v44 = (_QWORD *)v42[1].m128i_i64[0];
          v45 = *(_BYTE *)(v21 + 136);
        }
        v63 = v44;
        v46 = sub_73A830(v45 == 1, 5u);
        v63[2] = v46;
        for ( j = *(_QWORD *)(v21 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v48 = *(_QWORD *)(j + 128);
        sub_72BBE0((__int64)v66, v48, byte_4F06A51[0]);
        v46[2] = sub_73A720(v66, v48);
        sub_724E30((__int64)&v66);
        v64 = v46[2];
        v49 = sub_73A830((*(_BYTE *)(v21 + 156) & 4) != 0, 5u);
        *(_QWORD *)(v64 + 16) = v49;
        v49[2] = sub_73A830(0, 5u);
        v50 = sub_72BA30(5u);
        v51 = &qword_4F18A68;
        v52 = "__cudaRegisterManagedVariable";
        if ( (*(_WORD *)(v21 + 156) & 0x101) != 0x101 )
        {
          v51 = &qword_4F18A70;
          v52 = "__cudaRegisterVariable";
        }
        v53 = sub_7F8900(v52, (__m128i **)v51, (__int64)v50, v42);
        v11 = v67;
        v54 = unk_4F189C4;
        unk_4F189C4 = 1;
        sub_7E69E0(v53, v67);
        unk_4F189C4 = v54;
      }
      v21 = *(_QWORD *)(v21 + 112);
    }
    while ( v21 );
  }
  v23 = "____cudaRegisterLinkedBinary";
  sub_7FB010((__int64)v57, v65, (__int64)v68);
  v24 = sub_7F7930(0, 0, "__sti____cudaRegisterAll", (__int64)v67, (int *)&v66, (__int64)v68, 1);
  *(_BYTE *)(v24[4] + 197) |= 0x60u;
  dword_4D03EB8[0] = 1;
  if ( !qword_4D045BC )
    v23 = "__cudaRegisterBinary";
  v25 = (__m128i *)sub_731330((__int64)v56);
  v26 = sub_72BA30(5u);
  v27 = sub_7F8900(v23, (__m128i **)&qword_4F18A50, (__int64)v26, v25);
  sub_7E69E0(v27, v67);
  sub_7FB010((__int64)v24, (unsigned int)v66, (__int64)v68);
  dword_4D03EB8[0] = 0;
  if ( dword_4F077BC )
  {
    do
    {
      v29 = *(_QWORD **)(qword_4F07288 + 192);
      if ( !v29 )
        break;
      while ( 1 )
      {
        v30 = *(unsigned __int16 *)(v29[1] + 158LL);
        if ( (_WORD)v30 )
          break;
        v29 = (_QWORD *)*v29;
        if ( !v29 )
          return sub_801880(0);
      }
      sub_801880(v30);
    }
    while ( dword_4F077BC );
    return sub_801880(0);
  }
  else
  {
    sub_801880(0);
    *(_QWORD *)(v55 + 192) = 0;
    return v55;
  }
}
