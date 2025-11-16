// Function: sub_A3DDF0
// Address: 0xa3ddf0
//
void __fastcall sub_A3DDF0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        _BYTE *a8)
{
  __int64 v8; // rsi
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rbx
  char v13; // al
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // edx
  int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 v20; // r15
  __int64 *v21; // r14
  __int64 v22; // rax
  int v23; // edx
  unsigned int v24; // ecx
  __int64 v25; // r8
  unsigned int v26; // r9d
  unsigned __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // r14
  __int64 v30; // rsi
  int v31; // edx
  int v32; // edx
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // r10
  __int64 v36; // rsi
  __int64 *v37; // rax
  __int64 v38; // r11
  unsigned int v39; // eax
  unsigned int v40; // edx
  bool v41; // al
  int v42; // eax
  int v43; // eax
  int v44; // eax
  int v45; // ecx
  int v46; // r9d
  int v47; // r10d
  __int64 v48; // rbx
  __int128 v49; // xmm0
  __int64 v50; // r12
  __int64 v51; // r9
  __int64 *v52; // rbx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // rax
  int v56; // edx
  bool v57; // zf
  int v58; // edx
  int v59; // eax
  __int64 v60; // [rsp+0h] [rbp-B0h]
  _BYTE *v61; // [rsp+8h] [rbp-A8h]
  __int64 v62; // [rsp+10h] [rbp-A0h]
  __int64 v63; // [rsp+18h] [rbp-98h]
  __int64 v64; // [rsp+20h] [rbp-90h]
  __int64 *v65; // [rsp+28h] [rbp-88h]
  __int64 *v66; // [rsp+30h] [rbp-80h]
  unsigned int v67; // [rsp+3Ch] [rbp-74h]
  unsigned int v68; // [rsp+3Ch] [rbp-74h]
  _BYTE *v69; // [rsp+50h] [rbp-60h]
  __int128 v70; // [rsp+60h] [rbp-50h] BYREF
  _BYTE *v71; // [rsp+70h] [rbp-40h]

  v65 = a2;
  v8 = (__int64)a2 - a1;
  v64 = a3;
  if ( v8 <= 256 )
    return;
  v10 = v8;
  if ( !a3 )
  {
    v66 = v65;
    goto LABEL_48;
  }
  v63 = a7;
  v60 = a1 + 16;
  v61 = a8;
  v62 = *((_QWORD *)&a7 + 1);
  while ( 2 )
  {
    v11 = *(_QWORD *)(a1 + 16);
    --v64;
    *(_QWORD *)&v70 = v63;
    v12 = a1 + 16 * (v10 >> 5);
    *((_QWORD *)&v70 + 1) = v62;
    v71 = v61;
    v13 = sub_A3D900((__int64)&v70, v11, *(_QWORD *)v12);
    v14 = *(v65 - 2);
    if ( !v13 )
    {
      if ( !(unsigned __int8)sub_A3D900((__int64)&v70, *(_QWORD *)(a1 + 16), v14) )
      {
        v57 = (unsigned __int8)sub_A3D900((__int64)&v70, *(_QWORD *)v12, *(v65 - 2)) == 0;
        v15 = *(_QWORD *)a1;
        if ( !v57 )
        {
          *(_QWORD *)a1 = *(v65 - 2);
          v58 = *((_DWORD *)v65 - 2);
          *(v65 - 2) = v15;
          v59 = *(_DWORD *)(a1 + 8);
          *(_DWORD *)(a1 + 8) = v58;
          *((_DWORD *)v65 - 2) = v59;
          v18 = *(_QWORD *)(a1 + 16);
          v19 = *(_QWORD *)a1;
          goto LABEL_8;
        }
        goto LABEL_7;
      }
LABEL_42:
      v18 = *(_QWORD *)a1;
      v19 = *(_QWORD *)(a1 + 16);
      v44 = *(_DWORD *)(a1 + 8);
      v45 = *(_DWORD *)(a1 + 24);
      *(_QWORD *)a1 = v19;
      *(_QWORD *)(a1 + 16) = v18;
      *(_DWORD *)(a1 + 8) = v45;
      *(_DWORD *)(a1 + 24) = v44;
      goto LABEL_8;
    }
    if ( !(unsigned __int8)sub_A3D900((__int64)&v70, *(_QWORD *)v12, v14) )
    {
      if ( (unsigned __int8)sub_A3D900((__int64)&v70, *(_QWORD *)(a1 + 16), *(v65 - 2)) )
      {
        v55 = *(_QWORD *)a1;
        *(_QWORD *)a1 = *(v65 - 2);
        v56 = *((_DWORD *)v65 - 2);
        *(v65 - 2) = v55;
        LODWORD(v55) = *(_DWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 8) = v56;
        *((_DWORD *)v65 - 2) = v55;
        v18 = *(_QWORD *)(a1 + 16);
        v19 = *(_QWORD *)a1;
        goto LABEL_8;
      }
      goto LABEL_42;
    }
    v15 = *(_QWORD *)a1;
LABEL_7:
    *(_QWORD *)a1 = *(_QWORD *)v12;
    v16 = *(_DWORD *)(v12 + 8);
    *(_QWORD *)v12 = v15;
    v17 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v16;
    *(_DWORD *)(v12 + 8) = v17;
    v18 = *(_QWORD *)(a1 + 16);
    v19 = *(_QWORD *)a1;
LABEL_8:
    v20 = v60;
    v21 = v65;
    *(_QWORD *)&v70 = v63;
    *((_QWORD *)&v70 + 1) = v62;
    v71 = v61;
    while ( 1 )
    {
      v66 = (__int64 *)v20;
      if ( !(unsigned __int8)sub_A3D900((__int64)&v70, v18, v19) )
        break;
LABEL_13:
      v19 = *(_QWORD *)a1;
      v18 = *(_QWORD *)(v20 + 16);
      v20 += 16LL;
    }
    v27 = (unsigned __int64)(v21 - 2);
    v28 = *(_QWORD *)a1;
    v29 = *(v21 - 2);
    if ( *(_QWORD *)a1 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v28 + 24);
        v31 = *(_DWORD *)(v70 + 24);
        v25 = *(_QWORD *)(v70 + 8);
        if ( !v31 )
          goto LABEL_27;
        v32 = v31 - 1;
        v33 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v34 = (__int64 *)(v25 + 16LL * v33);
        v35 = *v34;
        if ( v30 == *v34 )
        {
LABEL_18:
          v26 = *((_DWORD *)v34 + 2);
          v36 = *(_QWORD *)(v29 + 24);
        }
        else
        {
          v42 = 1;
          while ( v35 != -4096 )
          {
            v46 = v42 + 1;
            v33 = v32 & (v42 + v33);
            v34 = (__int64 *)(v25 + 16LL * v33);
            v35 = *v34;
            if ( v30 == *v34 )
              goto LABEL_18;
            v42 = v46;
          }
          v36 = *(_QWORD *)(v29 + 24);
          v26 = 0;
        }
        v24 = v32 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v37 = (__int64 *)(v25 + 16LL * v24);
        v38 = *v37;
        if ( *v37 == v36 )
        {
LABEL_20:
          v39 = *((_DWORD *)v37 + 2);
          if ( v39 > v26 )
          {
            if ( v39 > **((_DWORD **)&v70 + 1) || *v71 )
              break;
            goto LABEL_24;
          }
        }
        else
        {
          v43 = 1;
          while ( v38 != -4096 )
          {
            v47 = v43 + 1;
            v24 = v32 & (v43 + v24);
            v37 = (__int64 *)(v25 + 16LL * v24);
            v38 = *v37;
            if ( *v37 == v36 )
              goto LABEL_20;
            v43 = v47;
          }
          v39 = 0;
        }
        v40 = **((_DWORD **)&v70 + 1);
        if ( v39 >= v26 )
        {
          if ( v26 > v40 )
            goto LABEL_28;
LABEL_27:
          if ( *v71 )
          {
LABEL_28:
            v67 = sub_BD2910(v28);
            v41 = v67 > (unsigned int)sub_BD2910(v29);
          }
          else
          {
            v68 = sub_BD2910(v28);
            v41 = v68 < (unsigned int)sub_BD2910(v29);
          }
          if ( !v41 )
            break;
          v28 = *(_QWORD *)a1;
          goto LABEL_24;
        }
        if ( v26 <= v40 && !*v71 )
          break;
LABEL_24:
        v29 = *(_QWORD *)(v27 - 16);
        v27 -= 16LL;
      }
      while ( v28 != v29 );
    }
    if ( v20 < v27 )
    {
      v22 = *(_QWORD *)v20;
      v21 = (__int64 *)v27;
      *(_QWORD *)v20 = *(_QWORD *)v27;
      v23 = *(_DWORD *)(v27 + 8);
      *(_QWORD *)v27 = v22;
      LODWORD(v22) = *(_DWORD *)(v20 + 8);
      *(_DWORD *)(v20 + 8) = v23;
      *(_DWORD *)(v27 + 8) = v22;
      goto LABEL_13;
    }
    v10 = v20 - a1;
    sub_A3DDF0(v20, (_DWORD)v65, v64, v24, v25, v26, a7, (__int64)a8);
    if ( (__int64)(v20 - a1) > 256 )
    {
      if ( v64 )
      {
        v65 = (__int64 *)v20;
        continue;
      }
LABEL_48:
      v48 = v10 >> 4;
      v49 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
      v50 = (v48 - 2) >> 1;
      v71 = a8;
      v69 = a8;
      v70 = v49;
      sub_A3DAC0(a1, v50, v48, *(_QWORD *)(a1 + 16 * v50), *(_QWORD *)(a1 + 16 * v50 + 8), a6, v49, (__int64)a8);
      do
      {
        --v50;
        sub_A3DAC0(a1, v50, v48, *(_QWORD *)(a1 + 16 * v50), *(_QWORD *)(a1 + 16 * v50 + 8), v51, v70, (__int64)v71);
      }
      while ( v50 );
      v52 = v66;
      do
      {
        v52 -= 2;
        v53 = *v52;
        v54 = v52[1];
        *v52 = *(_QWORD *)a1;
        *((_DWORD *)v52 + 2) = *(_DWORD *)(a1 + 8);
        sub_A3DAC0(a1, 0, ((__int64)v52 - a1) >> 4, v53, v54, v51, v49, (__int64)v69);
      }
      while ( (__int64)v52 - a1 > 16 );
    }
    break;
  }
}
