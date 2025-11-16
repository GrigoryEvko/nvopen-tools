// Function: sub_A5B850
// Address: 0xa5b850
//
void __fastcall sub_A5B850(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rax
  char v11; // si
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // r8
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 *v17; // r15
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rdi
  _BYTE *v30; // rax
  __int64 v31; // rdi
  _WORD *v32; // rdx
  __m128i *v33; // rdx
  __m128i si128; // xmm0
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rdi
  _WORD *v38; // rdx
  __int64 v39; // [rsp+0h] [rbp-70h]
  _QWORD *v40; // [rsp+8h] [rbp-68h]
  _QWORD *v41; // [rsp+8h] [rbp-68h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  _QWORD v44[10]; // [rsp+20h] [rbp-50h] BYREF

  if ( *(char *)(a2 + 7) < 0 )
  {
    v3 = sub_BD2BC0(a2);
    v5 = v3 + v4;
    v6 = 0;
    if ( *(char *)(a2 + 7) < 0 )
      v6 = sub_BD2BC0(a2);
    if ( (unsigned int)((v5 - v6) >> 4) )
    {
      sub_904010(*a1, " [ ");
      if ( *(char *)(a2 + 7) < 0 )
      {
        v7 = sub_BD2BC0(a2);
        v9 = v7 + v8;
        if ( *(char *)(a2 + 7) >= 0 )
          v10 = v9 >> 4;
        else
          LODWORD(v10) = (v9 - sub_BD2BC0(a2)) >> 4;
        if ( (_DWORD)v10 )
        {
          v11 = 1;
          v43 = 0;
          v39 = 16LL * (unsigned int)v10;
          while ( 1 )
          {
            v12 = 0;
            if ( *(char *)(a2 + 7) < 0 )
              v12 = sub_BD2BC0(a2);
            v13 = v43 + v12;
            v14 = *(_QWORD **)v13;
            v15 = 32LL * *(unsigned int *)(v13 + 8);
            v16 = 32LL * *(unsigned int *)(v13 + 12) - v15;
            v17 = (__int64 *)(a2 + v15 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
            if ( v11 )
              goto LABEL_13;
            v37 = *a1;
            v38 = *(_WORD **)(*a1 + 32);
            if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v38 <= 1u )
              break;
            *v38 = 8236;
            *(_QWORD *)(v37 + 32) += 2LL;
            v18 = *a1;
            v19 = *(_BYTE **)(*a1 + 32);
            if ( (unsigned __int64)v19 < *(_QWORD *)(*a1 + 24) )
            {
LABEL_14:
              *(_QWORD *)(v18 + 32) = v19 + 1;
              *v19 = 34;
              goto LABEL_15;
            }
LABEL_38:
            v40 = v14;
            sub_CB5D20(v18, 34);
            v14 = v40;
LABEL_15:
            sub_C92400(v14 + 2, *v14, *a1);
            v20 = *a1;
            v21 = *(_BYTE **)(*a1 + 32);
            if ( (unsigned __int64)v21 >= *(_QWORD *)(*a1 + 24) )
            {
              sub_CB5D20(v20, 34);
              v22 = *a1;
              v23 = *(_BYTE **)(*a1 + 32);
              if ( (unsigned __int64)v23 < *(_QWORD *)(*a1 + 24) )
              {
LABEL_17:
                *(_QWORD *)(v22 + 32) = v23 + 1;
                *v23 = 40;
                goto LABEL_18;
              }
            }
            else
            {
              *(_QWORD *)(v20 + 32) = v21 + 1;
              *v21 = 34;
              v22 = *a1;
              v23 = *(_BYTE **)(*a1 + 32);
              if ( (unsigned __int64)v23 < *(_QWORD *)(*a1 + 24) )
                goto LABEL_17;
            }
            sub_CB5D20(v22, 40);
LABEL_18:
            v24 = a1[1];
            v25 = a1[4];
            v26 = (__int64 *)((char *)v17 + v16);
            v44[1] = a1 + 5;
            v44[0] = off_4979428;
            v44[2] = v25;
            v44[3] = v24;
            if ( v26 != v17 )
            {
              while ( 1 )
              {
                v27 = *v17;
                v28 = *a1;
                if ( *v17 )
                {
LABEL_20:
                  sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(v27 + 8), v28);
                  v29 = *a1;
                  v30 = *(_BYTE **)(*a1 + 32);
                  if ( *(_BYTE **)(*a1 + 24) == v30 )
                  {
                    sub_CB6200(v29, " ", 1);
                  }
                  else
                  {
                    *v30 = 32;
                    ++*(_QWORD *)(v29 + 32);
                  }
                  sub_A5A730(*a1, *v17, (__int64)v44);
                  goto LABEL_23;
                }
                while ( 1 )
                {
                  v33 = *(__m128i **)(v28 + 32);
                  if ( *(_QWORD *)(v28 + 24) - (_QWORD)v33 <= 0x15u )
                  {
                    sub_CB6200(v28, "<null operand bundle!>", 22);
LABEL_23:
                    v17 += 4;
                    if ( v26 == v17 )
                      goto LABEL_28;
                  }
                  else
                  {
                    si128 = _mm_load_si128((const __m128i *)&xmmword_3F24B00);
                    v17 += 4;
                    v33[1].m128i_i32[0] = 1701602414;
                    v33[1].m128i_i16[2] = 15905;
                    *v33 = si128;
                    *(_QWORD *)(v28 + 32) += 22LL;
                    if ( v26 == v17 )
                      goto LABEL_28;
                  }
                  v31 = *a1;
                  v32 = *(_WORD **)(*a1 + 32);
                  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v32 <= 1u )
                    break;
                  *v32 = 8236;
                  *(_QWORD *)(v31 + 32) += 2LL;
                  v27 = *v17;
                  v28 = *a1;
                  if ( *v17 )
                    goto LABEL_20;
                }
                sub_CB6200(v31, ", ", 2);
              }
            }
LABEL_28:
            v35 = *a1;
            v36 = *(_BYTE **)(*a1 + 32);
            if ( (unsigned __int64)v36 >= *(_QWORD *)(*a1 + 24) )
            {
              sub_CB5D20(v35, 41);
            }
            else
            {
              *(_QWORD *)(v35 + 32) = v36 + 1;
              *v36 = 41;
            }
            v43 += 16;
            v11 = 0;
            if ( v43 == v39 )
              goto LABEL_31;
          }
          v41 = *(_QWORD **)v13;
          sub_CB6200(v37, ", ", 2);
          v14 = v41;
LABEL_13:
          v18 = *a1;
          v19 = *(_BYTE **)(*a1 + 32);
          if ( (unsigned __int64)v19 < *(_QWORD *)(*a1 + 24) )
            goto LABEL_14;
          goto LABEL_38;
        }
      }
LABEL_31:
      sub_904010(*a1, " ]");
    }
  }
}
