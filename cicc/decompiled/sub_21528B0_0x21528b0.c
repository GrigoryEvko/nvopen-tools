// Function: sub_21528B0
// Address: 0x21528b0
//
void __fastcall sub_21528B0(__int64 a1, __int64 a2, int a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rbx
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rsi
  int v14; // eax
  unsigned int v15; // eax
  int v16; // edx
  unsigned int v17; // eax
  unsigned int v18; // eax
  int v19; // edx
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // xmm0_8
  char *v25; // rdx
  __int64 v26; // rcx
  __int64 **v27; // rax
  unsigned int v28; // eax
  char v29; // di
  int v30; // edx
  unsigned __int8 v31; // al
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // esi
  unsigned int v36; // eax
  int v37; // edx
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  _QWORD *v40; // rax
  _QWORD *v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdi
  int v44; // ebx
  char v45; // r14
  unsigned int v46; // eax
  int v47; // ebx
  unsigned int v48; // eax
  int v49; // r14d
  int v50; // ecx
  _QWORD *v51; // rax
  unsigned int v52; // eax
  int v53; // edx
  _QWORD *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rax
  unsigned int v59; // eax
  int v60; // edx
  __int64 *v61; // rsi
  __int64 v62; // rbx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rdx
  char v66; // cl
  __int64 v67; // rax
  int v68; // edx
  unsigned int v69; // eax
  __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 *v73; // rdi
  char *v74; // rdx
  int v75; // ecx
  __int64 **v76; // rax
  unsigned int v77; // eax
  char v78; // si
  int v79; // edx
  __int64 v80; // rdx
  __int64 **v81; // rax
  __int64 v82; // rdx
  __int64 **v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 **v87; // rdx
  __int64 v88; // rax
  __int64 v89; // rcx
  int v90; // r8d
  int v91; // r9d
  unsigned int v92; // eax
  int v93; // edx
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 **v97; // rdx
  __int64 v98; // rax
  __int64 v99; // rcx
  int v100; // r8d
  int v101; // r9d
  unsigned int v102; // eax
  int v103; // edx
  __int64 v104; // [rsp+8h] [rbp-58h]
  __int64 v105; // [rsp+8h] [rbp-58h]
  int v106; // [rsp+8h] [rbp-58h]
  unsigned __int64 v107; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v108; // [rsp+18h] [rbp-48h]
  __int64 *v109; // [rsp+20h] [rbp-40h] BYREF
  _DWORD v110[14]; // [rsp+28h] [rbp-38h] BYREF

  v10 = sub_396DDB0();
  if ( *(_BYTE *)(a2 + 16) == 9 || sub_1593BB0(a2, a2, v8, v9) )
  {
    v14 = sub_12BE0A0(v10, *(_QWORD *)a2);
    if ( a3 < v14 )
      a3 = v14;
    if ( a3 > 0 )
    {
      v15 = *(_DWORD *)(a4 + 160);
      v16 = 0;
      do
      {
        ++v16;
        *(_BYTE *)(*(_QWORD *)(a4 + 8) + v15) = 0;
        v15 = *(_DWORD *)(a4 + 160) + 1;
        *(_DWORD *)(a4 + 160) = v15;
      }
      while ( a3 != v16 );
    }
  }
  else
  {
    v13 = *(_QWORD *)a2;
    switch ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) )
    {
      case 0:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 0xA:
      case 0xC:
      case 0xD:
      case 0xE:
      case 0x10:
        v17 = *(unsigned __int8 *)(a2 + 16);
        if ( v17 > 5 && (v17 == 11 || v17 <= 8 || v17 == 12) )
        {
          v47 = sub_12BE0A0(v10, v13);
          sub_2152590(a1, a2, a4);
          if ( a3 > v47 )
          {
            v48 = *(_DWORD *)(a4 + 160);
            v49 = a3 - v47;
            v50 = 0;
            do
            {
              ++v50;
              *(_BYTE *)(*(_QWORD *)(a4 + 8) + v48) = 0;
              v48 = *(_DWORD *)(a4 + 160) + 1;
              *(_DWORD *)(a4 + 160) = v48;
            }
            while ( v49 != v50 );
          }
        }
        else if ( a3 > 0 )
        {
          v18 = *(_DWORD *)(a4 + 160);
          v19 = 0;
          do
          {
            ++v19;
            *(_BYTE *)(*(_QWORD *)(a4 + 8) + v18) = 0;
            v18 = *(_DWORD *)(a4 + 160) + 1;
            *(_DWORD *)(a4 + 160) = v18;
          }
          while ( a3 != v19 );
        }
        return;
      case 1:
      case 2:
      case 3:
        v104 = *(_QWORD *)a2;
        if ( *(_BYTE *)(a2 + 16) != 14 )
          BUG();
        v20 = (_QWORD *)sub_16498A0(a2);
        if ( v104 == sub_1643290(v20) )
        {
          v61 = (__int64 *)(a2 + 32);
          if ( *(void **)(a2 + 32) == sub_16982C0() )
            sub_169D930((__int64)&v107, (__int64)v61);
          else
            sub_169D7E0((__int64)&v107, v61);
          sub_16A88B0((__int64)&v109, (__int64)&v107, 0x10u);
          LOWORD(v62) = (_WORD)v109;
          if ( v110[0] > 0x40u )
          {
            v62 = *v109;
            j_j___libc_free_0_0(v109);
          }
          v63 = *(unsigned int *)(a4 + 160);
          v64 = *(_QWORD *)(a4 + 8);
          LOWORD(v109) = v62;
          *(_BYTE *)(v64 + v63) = v62;
          v65 = *(_QWORD *)(a4 + 8);
          v66 = BYTE1(v109);
          v67 = (unsigned int)(*(_DWORD *)(a4 + 160) + 1);
          *(_DWORD *)(a4 + 160) = v67;
          *(_BYTE *)(v65 + v67) = v66;
          v68 = 2;
          v69 = *(_DWORD *)(a4 + 160) + 1;
          *(_DWORD *)(a4 + 160) = v69;
          if ( a3 > 2 )
          {
            do
            {
              ++v68;
              *(_BYTE *)(*(_QWORD *)(a4 + 8) + v69) = 0;
              v69 = *(_DWORD *)(a4 + 160) + 1;
              *(_DWORD *)(a4 + 160) = v69;
            }
            while ( a3 != v68 );
          }
          goto LABEL_80;
        }
        v21 = (_QWORD *)sub_16498A0(a2);
        if ( v104 == sub_16432A0(v21) )
        {
          v73 = (__int64 *)(a2 + 32);
          if ( *(void **)(a2 + 32) == sub_16982C0() )
            v73 = (__int64 *)(*(_QWORD *)(a2 + 40) + 8LL);
          *(float *)a5.m128i_i32 = sub_169D890(v73);
          v74 = (char *)&v109;
          v75 = _mm_cvtsi128_si32(a5);
          v76 = &v109;
          do
          {
            *(_BYTE *)v76 = v75;
            v76 = (__int64 **)((char *)v76 + 1);
            v75 >>= 8;
          }
          while ( (__int64 **)((char *)&v109 + 4) != v76 );
          v77 = *(_DWORD *)(a4 + 160);
          do
          {
            v78 = *v74++;
            *(_BYTE *)(*(_QWORD *)(a4 + 8) + v77) = v78;
            v77 = *(_DWORD *)(a4 + 160) + 1;
            *(_DWORD *)(a4 + 160) = v77;
          }
          while ( (char *)&v109 + 4 != v74 );
          if ( a3 > 4 )
          {
            v79 = 4;
            do
            {
              ++v79;
              *(_BYTE *)(*(_QWORD *)(a4 + 8) + v77) = 0;
              v77 = *(_DWORD *)(a4 + 160) + 1;
              *(_DWORD *)(a4 + 160) = v77;
            }
            while ( a3 != v79 );
          }
        }
        else
        {
          v22 = (_QWORD *)sub_16498A0(a2);
          sub_16432B0(v22);
          v23 = (__int64 *)(a2 + 32);
          if ( *(void **)(a2 + 32) == sub_16982C0() )
            v23 = (__int64 *)(*(_QWORD *)(a2 + 40) + 8LL);
          *(double *)&v24 = sub_169D8E0(v23);
          v25 = (char *)&v109;
          v26 = v24;
          v27 = &v109;
          do
          {
            *(_BYTE *)v27 = v26;
            v27 = (__int64 **)((char *)v27 + 1);
            v26 >>= 8;
          }
          while ( v110 != (_DWORD *)v27 );
          v28 = *(_DWORD *)(a4 + 160);
          do
          {
            v29 = *v25++;
            *(_BYTE *)(*(_QWORD *)(a4 + 8) + v28) = v29;
            v28 = *(_DWORD *)(a4 + 160) + 1;
            *(_DWORD *)(a4 + 160) = v28;
          }
          while ( v110 != (_DWORD *)v25 );
          if ( a3 > 8 )
          {
            v30 = 8;
            do
            {
              ++v30;
              *(_BYTE *)(*(_QWORD *)(a4 + 8) + v28) = 0;
              v28 = *(_DWORD *)(a4 + 160) + 1;
              *(_DWORD *)(a4 + 160) = v28;
            }
            while ( a3 != v30 );
          }
        }
        return;
      case 0xB:
        v105 = *(_QWORD *)a2;
        v38 = (_QWORD *)sub_16498A0(a2);
        if ( v105 == sub_1643330(v38) )
        {
          v51 = *(_QWORD **)(a2 + 24);
          if ( *(_DWORD *)(a2 + 32) > 0x40u )
            v51 = (_QWORD *)*v51;
          *(_BYTE *)(*(_QWORD *)(a4 + 8) + *(unsigned int *)(a4 + 160)) = (_BYTE)v51;
          v52 = *(_DWORD *)(a4 + 160) + 1;
          *(_DWORD *)(a4 + 160) = v52;
          if ( a3 > 1 )
          {
            v53 = 1;
            do
            {
              ++v53;
              *(_BYTE *)(*(_QWORD *)(a4 + 8) + v52) = 0;
              v52 = *(_DWORD *)(a4 + 160) + 1;
              *(_DWORD *)(a4 + 160) = v52;
            }
            while ( a3 != v53 );
          }
        }
        else
        {
          v39 = (_QWORD *)sub_16498A0(a2);
          if ( v105 == sub_1643340(v39) )
          {
            v54 = *(_QWORD **)(a2 + 24);
            if ( *(_DWORD *)(a2 + 32) > 0x40u )
              v54 = (_QWORD *)*v54;
            v55 = *(unsigned int *)(a4 + 160);
            v56 = *(_QWORD *)(a4 + 8);
            LOWORD(v109) = (_WORD)v54;
            *(_BYTE *)(v56 + v55) = (_BYTE)v54;
            v57 = *(_QWORD *)(a4 + 8);
            LOBYTE(v56) = BYTE1(v109);
            v58 = (unsigned int)(*(_DWORD *)(a4 + 160) + 1);
            *(_DWORD *)(a4 + 160) = v58;
            *(_BYTE *)(v57 + v58) = v56;
            v59 = *(_DWORD *)(a4 + 160) + 1;
            *(_DWORD *)(a4 + 160) = v59;
            if ( a3 > 2 )
            {
              v60 = 2;
              do
              {
                ++v60;
                *(_BYTE *)(*(_QWORD *)(a4 + 8) + v59) = 0;
                v59 = *(_DWORD *)(a4 + 160) + 1;
                *(_DWORD *)(a4 + 160) = v59;
              }
              while ( a3 != v60 );
            }
          }
          else
          {
            v40 = (_QWORD *)sub_16498A0(a2);
            if ( v105 == sub_1643350(v40) )
            {
              if ( *(_BYTE *)(a2 + 16) == 13 )
              {
                v80 = *(_QWORD *)(a2 + 24);
                if ( *(_DWORD *)(a2 + 32) > 0x40u )
                  v80 = *(_QWORD *)v80;
                v80 = (int)v80;
                v81 = &v109;
                do
                {
                  *(_BYTE *)v81 = v80;
                  v81 = (__int64 **)((char *)v81 + 1);
                  v80 >>= 8;
                }
                while ( (__int64 **)((char *)&v109 + 4) != v81 );
              }
              else
              {
                v84 = sub_14DBA30(a2, v10, 0);
                v85 = v84;
                if ( !v84 || *(_BYTE *)(v84 + 16) != 13 )
                {
                  v88 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
                  sub_214CD20(a4, v88, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v89, v90, v91);
                  v92 = *(_DWORD *)(a4 + 160);
                  v93 = 4;
                  do
                  {
                    *(_BYTE *)(*(_QWORD *)(a4 + 8) + v92) = 0;
                    v92 = *(_DWORD *)(a4 + 160) + 1;
                    *(_DWORD *)(a4 + 160) = v92;
                    --v93;
                  }
                  while ( v93 );
                  return;
                }
                v86 = *(_QWORD *)(v84 + 24);
                if ( *(_DWORD *)(v85 + 32) > 0x40u )
                  v86 = *(_QWORD *)v86;
                v86 = (int)v86;
                v87 = &v109;
                do
                {
                  *(_BYTE *)v87 = v86;
                  v87 = (__int64 **)((char *)v87 + 1);
                  v86 >>= 8;
                }
                while ( (__int64 **)((char *)&v109 + 4) != v87 );
              }
              ((void (__fastcall *)(__int64, __int64 **, __int64, _QWORD))loc_214AE80)(a4, &v109, 4, (unsigned int)a3);
              return;
            }
            v41 = (_QWORD *)sub_16498A0(a2);
            if ( v105 == sub_1643360(v41) )
            {
              if ( *(_BYTE *)(a2 + 16) == 13 )
              {
                v82 = *(_QWORD *)(a2 + 24);
                if ( *(_DWORD *)(a2 + 32) > 0x40u )
                  v82 = *(_QWORD *)v82;
                v83 = &v109;
                do
                {
                  *(_BYTE *)v83 = v82;
                  v83 = (__int64 **)((char *)v83 + 1);
                  v82 >>= 8;
                }
                while ( v110 != (_DWORD *)v83 );
              }
              else
              {
                v94 = sub_14DBA30(a2, v10, 0);
                v95 = v94;
                if ( !v94 || *(_BYTE *)(v94 + 16) != 13 )
                {
                  v98 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
                  sub_214CD20(a4, v98, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v99, v100, v101);
                  v102 = *(_DWORD *)(a4 + 160);
                  v103 = 8;
                  do
                  {
                    *(_BYTE *)(*(_QWORD *)(a4 + 8) + v102) = 0;
                    v102 = *(_DWORD *)(a4 + 160) + 1;
                    *(_DWORD *)(a4 + 160) = v102;
                    --v103;
                  }
                  while ( v103 );
                  return;
                }
                v96 = *(_QWORD *)(v94 + 24);
                if ( *(_DWORD *)(v95 + 32) > 0x40u )
                  v96 = *(_QWORD *)v96;
                v97 = &v109;
                do
                {
                  *(_BYTE *)v97 = v96;
                  v97 = (__int64 **)((char *)v97 + 1);
                  v96 >>= 8;
                }
                while ( v110 != (_DWORD *)v97 );
              }
              ((void (__fastcall *)(__int64, __int64 **, __int64, _QWORD))loc_214AE80)(a4, &v109, 8, (unsigned int)a3);
              return;
            }
            v42 = (_QWORD *)sub_16498A0(a2);
            sub_1643370(v42);
            v108 = *(_DWORD *)(a2 + 32);
            if ( v108 > 0x40 )
              sub_16A4FD0((__int64)&v107, (const void **)(a2 + 24));
            else
              v107 = *(_QWORD *)(a2 + 24);
            v43 = v10;
            v44 = 0;
            v106 = sub_12BE0A0(v43, *(_QWORD *)a2);
            if ( v106 )
            {
              do
              {
                sub_16A88B0((__int64)&v109, (__int64)&v107, 8u);
                v45 = (char)v109;
                if ( v110[0] > 0x40u )
                {
                  v45 = *(_BYTE *)v109;
                  j_j___libc_free_0_0(v109);
                }
                *(_BYTE *)(*(_QWORD *)(a4 + 8) + *(unsigned int *)(a4 + 160)) = v45;
                v46 = v108;
                ++*(_DWORD *)(a4 + 160);
                if ( v46 <= 0x40 )
                {
                  if ( v46 == 8 )
                    v107 = 0;
                  else
                    v107 >>= 8;
                }
                else
                {
                  sub_16A8110((__int64)&v107, 8u);
                }
                ++v44;
              }
              while ( v106 != v44 );
            }
LABEL_80:
            if ( v108 > 0x40 && v107 )
              j_j___libc_free_0_0(v107);
          }
        }
        return;
      case 0xF:
        v31 = *(_BYTE *)(a2 + 16);
        if ( v31 <= 3u )
        {
          v32 = *(unsigned int *)(a4 + 40);
          if ( (unsigned int)v32 >= *(_DWORD *)(a4 + 44) )
          {
            sub_16CD150(a4 + 32, (const void *)(a4 + 48), 0, 4, v11, v12);
            v32 = *(unsigned int *)(a4 + 40);
          }
          *(_DWORD *)(*(_QWORD *)(a4 + 32) + 4 * v32) = *(_DWORD *)(a4 + 160);
          v33 = *(unsigned int *)(a4 + 72);
          ++*(_DWORD *)(a4 + 40);
          if ( (unsigned int)v33 >= *(_DWORD *)(a4 + 76) )
          {
            sub_16CD150(a4 + 64, (const void *)(a4 + 80), 0, 8, v11, v12);
            v33 = *(unsigned int *)(a4 + 72);
          }
          *(_QWORD *)(*(_QWORD *)(a4 + 64) + 8 * v33) = a2;
          v34 = *(unsigned int *)(a4 + 120);
          ++*(_DWORD *)(a4 + 72);
          if ( (unsigned int)v34 < *(_DWORD *)(a4 + 124) )
            goto LABEL_35;
          goto LABEL_89;
        }
        if ( v31 == 5 )
        {
          v70 = sub_1649C60(a2);
          v71 = *(unsigned int *)(a4 + 40);
          if ( (unsigned int)v71 >= *(_DWORD *)(a4 + 44) )
          {
            sub_16CD150(a4 + 32, (const void *)(a4 + 48), 0, 4, v11, v12);
            v71 = *(unsigned int *)(a4 + 40);
          }
          *(_DWORD *)(*(_QWORD *)(a4 + 32) + 4 * v71) = *(_DWORD *)(a4 + 160);
          v72 = *(unsigned int *)(a4 + 72);
          ++*(_DWORD *)(a4 + 40);
          if ( (unsigned int)v72 >= *(_DWORD *)(a4 + 76) )
          {
            sub_16CD150(a4 + 64, (const void *)(a4 + 80), 0, 8, v11, v12);
            v72 = *(unsigned int *)(a4 + 72);
          }
          *(_QWORD *)(*(_QWORD *)(a4 + 64) + 8 * v72) = v70;
          v34 = *(unsigned int *)(a4 + 120);
          ++*(_DWORD *)(a4 + 72);
          if ( (unsigned int)v34 < *(_DWORD *)(a4 + 124) )
            goto LABEL_35;
LABEL_89:
          sub_16CD150(a4 + 112, (const void *)(a4 + 128), 0, 8, v11, v12);
          v34 = *(unsigned int *)(a4 + 120);
LABEL_35:
          *(_QWORD *)(*(_QWORD *)(a4 + 112) + 8 * v34) = a2;
          ++*(_DWORD *)(a4 + 120);
          ++*(_DWORD *)a4;
          v13 = *(_QWORD *)a2;
        }
        v35 = sub_12BE0A0(v10, v13);
        if ( v35 > 0 )
        {
          v36 = *(_DWORD *)(a4 + 160);
          v37 = 0;
          do
          {
            ++v37;
            *(_BYTE *)(*(_QWORD *)(a4 + 8) + v36) = 0;
            v36 = *(_DWORD *)(a4 + 160) + 1;
            *(_DWORD *)(a4 + 160) = v36;
          }
          while ( v35 != v37 );
        }
        break;
    }
  }
}
