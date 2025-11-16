// Function: sub_2F405D0
// Address: 0x2f405d0
//
__int64 __fastcall sub_2F405D0(__int64 a1, __int64 a2, unsigned int a3, unsigned __int8 a4, float *a5, __int64 a6)
{
  __int64 v8; // rbx
  int v9; // r8d
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  unsigned int v19; // esi
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r15
  __int64 v26; // r14
  unsigned int v27; // esi
  _DWORD *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  _DWORD *v31; // rdx
  bool v32; // dl
  unsigned __int8 v33; // r8
  float v34; // xmm1_4
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 *v38; // rdi
  __int64 v39; // r8
  __int64 v40; // rdx
  unsigned __int64 v41; // r11
  int *v42; // rcx
  int v43; // r10d
  __int64 v44; // rcx
  unsigned __int64 v45; // r8
  _DWORD *v46; // rdx
  unsigned int v47; // edi
  __int64 v48; // r8
  _DWORD *v49; // [rsp+8h] [rbp-78h]
  bool v50; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  int v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int16 *v54; // [rsp+20h] [rbp-60h]
  unsigned int v55; // [rsp+28h] [rbp-58h]
  bool v56; // [rsp+2Eh] [rbp-52h]
  unsigned int v58; // [rsp+38h] [rbp-48h]
  float v59; // [rsp+3Ch] [rbp-44h]
  unsigned int v62; // [rsp+4Ch] [rbp-34h]

  v8 = a1;
  v9 = sub_2E21680(*(_QWORD **)(a1 + 24), a2, a3);
  result = 0;
  if ( v9 <= 1 )
  {
    v56 = 1;
    if ( *(_DWORD *)(a2 + 8) )
      v56 = sub_2E13500(*(_QWORD *)(a1 + 32), a2) != 0;
    v11 = *(_QWORD *)(a1 + 16);
    v58 = *(_DWORD *)(*(_QWORD *)(v11 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 4);
    if ( !v58 )
      v58 = *(_DWORD *)(v11 + 952);
    v12 = *(_QWORD *)(a1 + 56);
    v62 = 0;
    v59 = 0.0;
    v54 = (__int16 *)(*(_QWORD *)(v12 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v12 + 8) + 24LL * a3 + 16) >> 12));
    v55 = *(_DWORD *)(*(_QWORD *)(v12 + 8) + 24LL * a3 + 16) & 0xFFF;
    while ( 1 )
    {
      if ( !v54 )
      {
LABEL_44:
        *(_DWORD *)a5 = v62;
        a5[1] = v59;
        return 1;
      }
      v13 = sub_2E21610(*(_QWORD *)(v8 + 24), a2, v55);
      v18 = v13;
      v19 = qword_5023488[8];
      if ( !*(_BYTE *)(v13 + 161) || (v20 = *(_DWORD *)(v13 + 120), LODWORD(qword_5023488[8]) < v20) )
      {
        sub_2E1AC90(v18, qword_5023488[8], v14, v15, v16, v17);
        v20 = *(_DWORD *)(v18 + 120);
        v19 = qword_5023488[8];
      }
      v21 = v20;
      if ( v20 >= v19 )
        return 0;
      v22 = *(_QWORD *)(v18 + 112);
      if ( v22 != v22 + 8LL * v20 )
      {
        v53 = *(_QWORD *)(v18 + 112);
        v23 = v8;
        v24 = v22 + 8 * v21;
        v25 = v23;
        while ( 1 )
        {
          v26 = *(_QWORD *)(v24 - 8);
          v27 = *(_DWORD *)(v26 + 112);
          if ( *(_QWORD *)(a6 + 120) )
          {
            v36 = *(_QWORD *)(a6 + 96);
            if ( v36 )
            {
              v37 = a6 + 88;
              do
              {
                if ( v27 > *(_DWORD *)(v36 + 32) )
                {
                  v36 = *(_QWORD *)(v36 + 24);
                }
                else
                {
                  v37 = v36;
                  v36 = *(_QWORD *)(v36 + 16);
                }
              }
              while ( v36 );
              if ( a6 + 88 != v37 && v27 >= *(_DWORD *)(v37 + 32) )
                return 0;
            }
          }
          else
          {
            v28 = *(_DWORD **)a6;
            v29 = *(_QWORD *)a6 + 4LL * *(unsigned int *)(a6 + 8);
            if ( *(_QWORD *)a6 != v29 )
            {
              while ( v27 != *v28 )
              {
                if ( (_DWORD *)v29 == ++v28 )
                  goto LABEL_24;
              }
              if ( (_DWORD *)v29 != v28 )
                return 0;
            }
          }
LABEL_24:
          v30 = v27 & 0x7FFFFFFF;
          v31 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 16) + 920LL) + 8 * v30);
          if ( *v31 == 6 )
            return 0;
          if ( INFINITY != *(float *)(a2 + 116) )
          {
            if ( v58 == v31[1] || v58 < v31[1] )
              return 0;
            v32 = 0;
            goto LABEL_31;
          }
          if ( INFINITY == *(float *)(v26 + 116) )
          {
            v38 = *(__int64 **)(v25 + 64);
            v39 = *(_QWORD *)(*(_QWORD *)(v25 + 48) + 56LL);
            v40 = *v38;
            v41 = *(_QWORD *)(v39 + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
            v42 = (int *)(*v38 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v41 + 24LL));
            v43 = *v42;
            if ( *((_DWORD *)v38 + 2) != *v42 )
            {
              v51 = *v38 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v41 + 24LL);
              sub_2F60630(
                v38,
                *(_QWORD *)(v39 + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                v40,
                v42);
              v38 = *(__int64 **)(v25 + 64);
              v42 = (int *)v51;
              v39 = *(_QWORD *)(*(_QWORD *)(v25 + 48) + 56LL);
              v43 = *((_DWORD *)v38 + 2);
              v40 = *v38;
              v30 = *(_DWORD *)(v26 + 112) & 0x7FFFFFFF;
            }
            v44 = (unsigned int)v42[1];
            v45 = *(_QWORD *)(v39 + 16 * v30) & 0xFFFFFFFFFFFFFFF8LL;
            v46 = (_DWORD *)(v40 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v45 + 24LL));
            if ( *v46 != v43 )
            {
              v49 = v46;
              v52 = v44;
              sub_2F60630(v38, v45, v46, v44);
              v46 = v49;
              LODWORD(v44) = v52;
              v30 = *(_DWORD *)(v26 + 112) & 0x7FFFFFFF;
            }
            v47 = v46[1];
            v48 = *(_QWORD *)(*(_QWORD *)(v25 + 16) + 920LL);
            v32 = v47 > (unsigned int)v44;
            if ( v58 == *(_DWORD *)(v48 + 8 * v30 + 4) )
              return 0;
            if ( v58 >= *(_DWORD *)(v48 + 8 * v30 + 4) )
              goto LABEL_31;
            if ( v47 <= (unsigned int)v44 )
              return 0;
          }
          else
          {
            if ( v58 == v31[1] )
              return 0;
            if ( v58 >= v31[1] )
              goto LABEL_30;
          }
          v62 += 10;
LABEL_30:
          v32 = 1;
LABEL_31:
          v50 = v32;
          v33 = sub_300C040(*(_QWORD *)(v25 + 40));
          v62 += v33;
          v34 = fmaxf(*(float *)(v26 + 116), v59);
          v59 = v34;
          if ( v62 >= *(_DWORD *)a5 && (v62 != *(_DWORD *)a5 || a5[1] <= v34)
            || !v50
            && (!sub_2F40590(v25, a2, a4, v26, v33)
             || *(_DWORD *)a5 != -1
             && v56
             && sub_2E13500(*(_QWORD *)(v25 + 32), v26)
             && (!*(_BYTE *)(v25 + 88) || !(unsigned __int8)sub_2F50850(v25, v26, a3))) )
          {
            return 0;
          }
          v24 -= 8;
          if ( v53 == v24 )
          {
            v8 = v25;
            break;
          }
        }
      }
      v35 = *v54;
      v55 += v35;
      ++v54;
      if ( !(_WORD)v35 )
        goto LABEL_44;
    }
  }
  return result;
}
