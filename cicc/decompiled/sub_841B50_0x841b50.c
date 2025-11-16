// Function: sub_841B50
// Address: 0x841b50
//
__int64 __fastcall sub_841B50(
        const __m128i *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        _DWORD *a6,
        _QWORD *a7,
        _QWORD *a8,
        _DWORD *a9)
{
  const __m128i *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // r9
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // rdx
  const __m128i *i; // rax
  __int64 v21; // r12
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 *v31; // r9
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r12
  int *v36; // rax
  char v37; // al
  _QWORD *v38; // r15
  __m128i *v39; // r13
  __m128i *v40; // rax
  unsigned int v41; // r12d
  _QWORD *v42; // r13
  _QWORD *v43; // rbx
  __int64 v44; // r13
  __int64 v45; // rdi
  _BYTE *v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // [rsp+Ch] [rbp-234h]
  __int64 v50; // [rsp+10h] [rbp-230h]
  int *v52; // [rsp+28h] [rbp-218h]
  int v53; // [rsp+38h] [rbp-208h]
  __m128i *v55; // [rsp+40h] [rbp-200h]
  __int64 v57; // [rsp+58h] [rbp-1E8h]
  unsigned int v58; // [rsp+60h] [rbp-1E0h]
  int v59; // [rsp+64h] [rbp-1DCh]
  int v60; // [rsp+68h] [rbp-1D8h]
  int v61; // [rsp+6Ch] [rbp-1D4h]
  int v62; // [rsp+74h] [rbp-1CCh] BYREF
  __int64 v63; // [rsp+78h] [rbp-1C8h] BYREF
  __int64 *v64; // [rsp+80h] [rbp-1C0h] BYREF
  _QWORD *v65; // [rsp+88h] [rbp-1B8h] BYREF
  _BYTE v66[32]; // [rsp+90h] [rbp-1B0h] BYREF
  _BYTE v67[400]; // [rsp+B0h] [rbp-190h] BYREF

  v9 = a1;
  v60 = a2;
  v59 = a3;
  v10 = sub_82BD70(a1, a2, a3);
  v15 = *(_QWORD *)(v10 + 1024);
  v16 = v10;
  if ( v15 == *(_QWORD *)(v10 + 1016) )
    sub_8332F0(v10, a2, v11, v12, v13, v14);
  v17 = *(_QWORD *)(v16 + 1008) + 40 * v15;
  if ( v17 )
  {
    *(_BYTE *)v17 &= 0xFCu;
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)(v17 + 16) = 0;
    *(_QWORD *)(v17 + 24) = 0;
    *(_QWORD *)(v17 + 32) = 0;
  }
  *(_QWORD *)(v16 + 1024) = v15 + 1;
  *a6 = 0;
  *(_DWORD *)a7 = 0;
  if ( a8 )
    *a8 = 0;
  *a9 = 0;
  if ( a1[8].m128i_i8[12] == 12 )
  {
    do
      v9 = (const __m128i *)v9[10].m128i_i64[0];
    while ( v9[8].m128i_i8[12] == 12 );
  }
  v18 = (__int64 *)v9;
  if ( (unsigned int)sub_8D23B0(v9) )
  {
    v18 = (__int64 *)v9;
    if ( (unsigned int)sub_8D3A70(v9) )
    {
      a2 = 0;
      v18 = (__int64 *)v9;
      sub_8AD220(v9, 0);
    }
  }
  for ( i = v9; i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
    ;
  v50 = *(_QWORD *)(i->m128i_i64[0] + 96);
  if ( (*(_BYTE *)(v50 + 177) & 0x30) == 0x20 || (v9[11].m128i_i8[1] & 0x20) != 0 )
  {
    v21 = 0;
    if ( (v60 & 0xFFFFFFFE) == 0 )
      *a9 = 1;
    goto LABEL_16;
  }
  v62 = 0;
  v65 = 0;
  v57 = sub_7D3790(0xFu, v9->m128i_i8);
  v53 = 2;
  v49 = -1;
  v58 = 0;
  v61 = 0;
  while ( 1 )
  {
    if ( v57 )
    {
      v35 = sub_82C1B0(v57, (__int64)&v65, (__int64)a8, (__int64)v66);
      if ( v35 )
      {
        v36 = 0;
        if ( !v61 )
          v36 = &v62;
        v52 = v36;
        while ( 1 )
        {
          while ( 1 )
          {
            v37 = *(_BYTE *)(v35 + 80);
            if ( v37 == 16 )
            {
              v35 = **(_QWORD **)(v35 + 88);
              v37 = *(_BYTE *)(v35 + 80);
            }
            if ( v37 == 24 )
            {
              v35 = *(_QWORD *)(v35 + 88);
              v37 = *(_BYTE *)(v35 + 80);
            }
            if ( (v37 == 20) != v61 )
              goto LABEL_42;
            if ( !v58 )
              break;
            if ( v37 != 20 )
              break;
            v44 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v35 + 88) + 176LL) + 152LL) + 168LL);
            if ( !v44
              || !(unsigned int)sub_8D3070(*(_QWORD *)(v44 + 8))
              && (v59 || !(unsigned int)sub_8D3110(*(_QWORD *)(v44 + 8))) )
            {
              break;
            }
            v45 = sub_8D46C0(*(_QWORD *)(v44 + 8));
            if ( (*(_BYTE *)(v45 + 140) & 0xFB) == 8 ? v49 & ~(unsigned int)sub_8D4C10(v45, dword_4F077C4 != 2) : v49 )
              break;
            v58 = 1;
LABEL_42:
            v35 = sub_82C230(v66);
            if ( !v35 )
              goto LABEL_43;
          }
          v38 = qword_4D03C60;
          if ( qword_4D03C60 )
            qword_4D03C60 = (_QWORD *)*qword_4D03C60;
          else
            v38 = (_QWORD *)sub_823970(104);
          sub_82D850((__int64)v38);
          v39 = (__m128i *)qword_4D03C60;
          if ( qword_4D03C60 )
            qword_4D03C60 = (_QWORD *)*qword_4D03C60;
          else
            v39 = (__m128i *)sub_823970(104);
          sub_82D850((__int64)v39);
          sub_83DAC0(v35, v9, v60, v59, v39, (__int64 *)&v64, &v63, v52);
          if ( v39->m128i_i32[2] == 7
            || (v55 = sub_82EAF0(v63, v57, 0),
                v40 = sub_73C570(v9, a4),
                sub_6EA0A0((__int64)v40, (__int64)v67),
                sub_8399C0((__int64)v67, 0, 0, v55, v63, (__int64)v38),
                *((_DWORD *)v38 + 2) == 7) )
          {
            *v38 = v39;
            sub_82D8A0(v38);
            sub_725130(v64);
            v35 = sub_82C230(v66);
            if ( !v35 )
              break;
          }
          else
          {
            *v38 = v39;
            if ( *(_BYTE *)(v35 + 80) != 20 )
            {
              sub_82B8E0(v35, v57, (__int64)v38, (__int64 *)&v65);
              if ( !*((_DWORD *)v38 + 2) && (*((_BYTE *)v38 + 84) & 2) == 0 )
              {
                v41 = v39->m128i_u32[2];
                if ( !v41 )
                {
                  if ( (unsigned int)sub_8D2FB0(v39[2].m128i_i64[0]) )
                  {
                    v48 = sub_8D46C0(v39[2].m128i_i64[0]);
                    if ( (*(_BYTE *)(v48 + 140) & 0xFB) == 8 )
                      v41 = sub_8D4C10(v48, dword_4F077C4 != 2) & v49 & ~v60;
                    v49 = v41;
                    v58 = 1;
                  }
                }
              }
              goto LABEL_42;
            }
            sub_82B9D0(v35, v57, v63, 0, (__int64)v64, (__int64)v38, (__int64 *)&v65);
            v35 = sub_82C230(v66);
            if ( !v35 )
              break;
          }
        }
      }
    }
LABEL_43:
    if ( v53 == 1 )
      break;
    v33 = v58;
    v61 = 1;
    v53 = 1;
    if ( v58 )
    {
      if ( !v49 )
        break;
    }
  }
  v18 = (__int64 *)&v65;
  sub_82D8D0((__int64 *)&v65, a5, a7, a6, v33, v34);
  v42 = v65;
  a2 = *(unsigned int *)a7;
  if ( (_DWORD)a2 || *a6 )
  {
    if ( v65 )
    {
      v21 = 0;
      goto LABEL_49;
    }
    goto LABEL_80;
  }
  if ( !v65 )
  {
    if ( (*(_BYTE *)(v50 + 177) & 0x20) != 0 )
      *a9 = 1;
LABEL_80:
    v21 = 0;
    goto LABEL_16;
  }
  v21 = v65[1];
  v47 = *(_BYTE **)(v21 + 88);
  if ( (v47[194] & 4) != 0 && (v47[206] & 0x10) == 0 && (v47[193] & 4) == 0 && (*(_BYTE *)(v50 + 177) & 0x20) != 0 )
  {
    v21 = 0;
    *a9 = 1;
  }
  do
  {
LABEL_49:
    v43 = v42;
    v42 = (_QWORD *)*v42;
    sub_725130((__int64 *)v43[5]);
    v18 = (__int64 *)v43[15];
    sub_82D8A0(v18);
    *v43 = qword_4D03C68;
    qword_4D03C68 = v43;
  }
  while ( v42 );
LABEL_16:
  v23 = sub_82BD70(v18, a2, v19);
  v26 = *(_QWORD *)(v23 + 1008);
  v27 = *(_QWORD *)(v26 + 8 * (5LL * *(_QWORD *)(v23 + 1024) - 5) + 32);
  if ( v27 )
  {
    sub_823A00(*(_QWORD *)v27, 16LL * (unsigned int)(*(_DWORD *)(v27 + 8) + 1), v26, v22, v24, v25);
    sub_823A00(v27, 16, v28, v29, v30, v31);
  }
  --*(_QWORD *)(v23 + 1024);
  return v21;
}
