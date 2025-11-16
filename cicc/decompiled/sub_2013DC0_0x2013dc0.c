// Function: sub_2013DC0
// Address: 0x2013dc0
//
__int64 __fastcall sub_2013DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  const __m128i *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __m128i *v11; // r8
  const __m128i *v12; // r9
  __m128i v13; // xmm0
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 i; // r14
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // r15
  int v24; // eax
  unsigned int v25; // r12d
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // r13
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 v36; // r13
  __int64 v38; // rdx
  char v39; // al
  __int64 *v40; // r13
  unsigned int v41; // ebx
  int v42; // r12d
  __int64 v43; // rdx
  __m128i *v44; // r8
  __int64 v45; // [rsp+18h] [rbp-138h]
  __int64 v46; // [rsp+20h] [rbp-130h]
  __int64 v47; // [rsp+28h] [rbp-128h]
  __m128i v48; // [rsp+30h] [rbp-120h] BYREF
  __int64 v49; // [rsp+40h] [rbp-110h]
  __int64 v50; // [rsp+48h] [rbp-108h]
  __int64 v51; // [rsp+50h] [rbp-100h]
  __int64 v52; // [rsp+58h] [rbp-F8h]
  __int64 v53; // [rsp+60h] [rbp-F0h]
  __int64 v54; // [rsp+68h] [rbp-E8h]
  __m128i v55; // [rsp+70h] [rbp-E0h]
  _BYTE v56[32]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD v57[3]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v58; // [rsp+B8h] [rbp-98h]
  __int16 v59; // [rsp+BAh] [rbp-96h]
  int v60; // [rsp+BCh] [rbp-94h]
  __int64 *v61; // [rsp+C0h] [rbp-90h]
  __int64 v62; // [rsp+C8h] [rbp-88h]
  __int64 v63; // [rsp+D0h] [rbp-80h]
  __int64 v64; // [rsp+D8h] [rbp-78h]
  int v65; // [rsp+E0h] [rbp-70h]
  __int64 v66; // [rsp+E8h] [rbp-68h]
  int v67; // [rsp+F0h] [rbp-60h]
  __int64 v68; // [rsp+F8h] [rbp-58h] BYREF
  __int64 v69; // [rsp+100h] [rbp-50h]
  _QWORD *v70; // [rsp+108h] [rbp-48h]
  __int64 v71; // [rsp+110h] [rbp-40h]
  __int64 v72; // [rsp+118h] [rbp-38h] BYREF

  v7 = a1;
  v8 = *(const __m128i **)(a1 + 8);
  v9 = v8[11].m128i_i64[0];
  v48 = _mm_loadu_si128(v8 + 11);
  v10 = sub_1D274F0(1u, a3, a4, a5, a6);
  v13 = _mm_load_si128(&v48);
  v72 = 0;
  v62 = v10;
  v14 = 212;
  v64 = 0x100000000LL;
  v70 = v57;
  v55 = v13;
  v63 = 0;
  v65 = 0;
  v71 = 0;
  v66 = 0;
  v67 = -65536;
  LODWORD(v69) = v13.m128i_i32[2];
  v68 = v13.m128i_i64[0];
  v15 = *(_QWORD *)(v9 + 48);
  memset(v57, 0, sizeof(v57));
  v58 = 212;
  v59 = 0;
  v72 = v15;
  if ( v15 )
    *(_QWORD *)(v15 + 24) = &v72;
  v71 = v9 + 48;
  *(_QWORD *)(v9 + 48) = &v68;
  v61 = &v68;
  v16 = *(_QWORD *)(a1 + 8);
  v54 = 0;
  v53 = 0;
  *(_QWORD *)(v16 + 176) = 0;
  v17 = (unsigned int)v54;
  LODWORD(v64) = 1;
  *(_DWORD *)(v16 + 184) = v54;
  v18 = *(_QWORD *)(a1 + 8);
  v60 = -2;
  v19 = *(_QWORD *)(v18 + 200);
  for ( i = v18 + 192; i != v19; v19 = *(_QWORD *)(v19 + 8) )
  {
    while ( 1 )
    {
      if ( !v19 )
        BUG();
      v17 = *(unsigned int *)(v19 + 48);
      if ( !(_DWORD)v17 )
        break;
      *(_DWORD *)(v19 + 20) = -2;
      v19 = *(_QWORD *)(v19 + 8);
      if ( i == v19 )
        goto LABEL_11;
    }
    *(_DWORD *)(v19 + 20) = 0;
    v21 = *(unsigned int *)(a1 + 1376);
    if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 1380) )
    {
      v14 = a1 + 1384;
      sub_16CD150(a1 + 1368, (const void *)(a1 + 1384), 0, 8, (int)v11, (int)v12);
      v21 = *(unsigned int *)(a1 + 1376);
    }
    v17 = *(_QWORD *)(a1 + 1368);
    *(_QWORD *)(v17 + 8 * v21) = v19 - 8;
    ++*(_DWORD *)(a1 + 1376);
  }
LABEL_11:
  v22 = *(unsigned int *)(a1 + 1376);
  if ( (_DWORD)v22 )
  {
    v48.m128i_i8[0] = 0;
    while ( 1 )
    {
      if ( byte_4FCEBE0 )
        sub_2010FB0(a1, v14, v17, v22, (__int64)v11, (__int64)v12);
      v22 = *(unsigned int *)(a1 + 1376);
      v17 = *(_QWORD *)(a1 + 1368);
      v23 = *(_QWORD *)(v17 + 8 * v22 - 8);
      --*(_DWORD *)(a1 + 1376);
      v24 = *(unsigned __int16 *)(v23 + 24);
      if ( v24 != 8 && v24 != 32 )
      {
        v25 = *(_DWORD *)(v23 + 60);
        if ( v25 )
          break;
      }
LABEL_21:
      if ( !*(_DWORD *)(v23 + 56) )
        goto LABEL_26;
      v22 = v6;
      v28 = 0;
      v29 = *(unsigned int *)(v23 + 56);
      while ( 1 )
      {
        v17 = *(_QWORD *)(v23 + 32) + 40 * v28;
        v14 = *(_QWORD *)v17;
        if ( *(_WORD *)(*(_QWORD *)v17 + 24LL) != 8 && *(_WORD *)(*(_QWORD *)v17 + 24LL) != 32 )
          break;
LABEL_24:
        if ( v29 == ++v28 )
        {
LABEL_25:
          v6 = v22;
          goto LABEL_26;
        }
      }
      v38 = *(_QWORD *)(v14 + 40) + 16LL * *(unsigned int *)(v17 + 8);
      LOBYTE(v22) = *(_BYTE *)v38;
      v14 = *(_QWORD *)a1;
      sub_1F40D10((__int64)v56, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v22, *(_QWORD *)(v38 + 8));
      switch ( v56[0] )
      {
        case 0:
          goto LABEL_24;
        case 1:
          v14 = v23;
          v6 = v22;
          v39 = sub_2141750(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 2:
          v14 = v23;
          v6 = v22;
          v39 = sub_2136BC0(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 3:
          v14 = v23;
          v6 = v22;
          v39 = sub_2124800(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 4:
          v14 = v23;
          v6 = v22;
          v39 = sub_211F5C0(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 5:
          v14 = v23;
          v6 = v22;
          v39 = sub_2035F80(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 6:
          v14 = v23;
          v6 = v22;
          v39 = sub_202E5A0(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 7:
          v14 = v23;
          v6 = v22;
          v39 = sub_2040000(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        case 8:
          v14 = v23;
          v6 = v22;
          v39 = sub_2126F30(a1, v23, (unsigned int)v28);
          v48.m128i_i8[0] = 1;
          break;
        default:
          goto LABEL_25;
      }
      if ( v39 )
      {
        *(_DWORD *)(v23 + 28) = -1;
        v14 = v23;
        v40 = sub_2010420(a1, v23, v17, v22, v11, v12);
        if ( (__int64 *)v23 == v40 || !*(_DWORD *)(v23 + 60) )
          goto LABEL_36;
        v46 = v6;
        v41 = 0;
        v42 = *(_DWORD *)(v23 + 60);
        do
        {
          v43 = v41;
          v44 = (__m128i *)v41;
          v14 = v23;
          ++v41;
          sub_2013400(a1, v23, v43, (__int64)v40, v44, v12);
        }
        while ( v41 != v42 );
        v6 = v46;
        if ( !*(_DWORD *)(a1 + 1376) )
        {
LABEL_37:
          v7 = a1;
          goto LABEL_38;
        }
      }
      else
      {
LABEL_26:
        v30 = *(_QWORD *)(v23 + 48);
        *(_DWORD *)(v23 + 28) = -3;
        if ( v30 )
        {
          v14 = a1 + 1384;
          while ( 1 )
          {
            v31 = *(_QWORD *)(v30 + 16);
            v32 = *(_DWORD *)(v31 + 28);
            if ( v32 > 0 )
              goto LABEL_32;
            if ( v32 == -1 )
            {
LABEL_28:
              v30 = *(_QWORD *)(v30 + 32);
              if ( !v30 )
                break;
            }
            else
            {
              v32 = *(_DWORD *)(v31 + 56);
LABEL_32:
              v17 = (unsigned int)(v32 - 1);
              *(_DWORD *)(v31 + 28) = v17;
              if ( v32 != 1 )
                goto LABEL_28;
              v33 = *(unsigned int *)(a1 + 1376);
              if ( (unsigned int)v33 >= *(_DWORD *)(a1 + 1380) )
              {
                v14 = a1 + 1384;
                sub_16CD150(a1 + 1368, (const void *)(a1 + 1384), 0, 8, (int)v11, (int)v12);
                v33 = *(unsigned int *)(a1 + 1376);
              }
              v17 = *(_QWORD *)(a1 + 1368);
              *(_QWORD *)(v17 + 8 * v33) = v31;
              ++*(_DWORD *)(a1 + 1376);
              v30 = *(_QWORD *)(v30 + 32);
              if ( !v30 )
                break;
            }
          }
        }
LABEL_36:
        if ( !*(_DWORD *)(a1 + 1376) )
          goto LABEL_37;
      }
    }
    v22 = v47;
    v45 = v25;
    v26 = 0;
    while ( 2 )
    {
      v14 = *(_QWORD *)a1;
      v27 = *(_QWORD *)(v23 + 40) + 16 * v26;
      LOBYTE(v22) = *(_BYTE *)v27;
      sub_1F40D10((__int64)v56, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v22, *(_QWORD *)(v27 + 8));
      v17 = v56[0];
      switch ( v56[0] )
      {
        case 1:
          v14 = v23;
          v47 = v22;
          sub_2143140(a1, v23, (unsigned int)v26);
          break;
        case 2:
          v14 = v23;
          v47 = v22;
          sub_2141F70(a1, v23, (unsigned int)v26);
          break;
        case 3:
          v14 = v23;
          v47 = v22;
          v48.m128i_i8[0] = sub_2126090(a1, v23, (unsigned int)v26);
          if ( !v48.m128i_i8[0] )
            goto LABEL_21;
          break;
        case 4:
          v14 = v23;
          v47 = v22;
          sub_211E010(a1, v23, (unsigned int)v26);
          break;
        case 5:
          v14 = v23;
          v47 = v22;
          sub_2036110(a1, v23, (unsigned int)v26);
          break;
        case 6:
          v14 = v23;
          v47 = v22;
          sub_2029C10(a1, v23, (unsigned int)v26);
          break;
        case 7:
          v14 = v23;
          v47 = v22;
          sub_20416B0(a1, v23, (unsigned int)v26);
          break;
        case 8:
          v14 = v23;
          v47 = v22;
          sub_21276B0(a1, v23, (unsigned int)v26);
          break;
        default:
          if ( v45 != ++v26 )
            continue;
          v47 = v22;
          goto LABEL_21;
      }
      break;
    }
    v48.m128i_i8[0] = 1;
    goto LABEL_26;
  }
  v48.m128i_i8[0] = 0;
LABEL_38:
  if ( byte_4FCEBE0 )
    sub_2010FB0(v7, v14, v17, v22, (__int64)v11, (__int64)v12);
  v34 = v68;
  v35 = *(_QWORD *)(v7 + 8);
  v36 = v69;
  if ( v68 )
  {
    nullsub_686();
    v52 = v36;
    v14 = 0;
    v51 = v34;
    *(_QWORD *)(v35 + 176) = v34;
    *(_DWORD *)(v35 + 184) = v52;
    sub_1D23870();
  }
  else
  {
    v50 = v69;
    v49 = 0;
    *(_QWORD *)(v35 + 176) = 0;
    *(_DWORD *)(v35 + 184) = v50;
  }
  sub_1D2D9C0(*(const __m128i **)(v7 + 8), v14, v17, v22, (__int64)v11, (__int64)v12);
  sub_1D189A0((__int64)v57);
  return v48.m128i_u8[0];
}
