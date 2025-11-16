// Function: sub_2E05630
// Address: 0x2e05630
//
__int64 __fastcall sub_2E05630(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rdx
  _DWORD *v7; // rax
  _DWORD *v8; // r12
  _DWORD *v9; // rbx
  unsigned int v10; // r14d
  __int32 v11; // r15d
  __int64 v12; // rax
  unsigned __int64 v13; // r8
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned int v17; // esi
  int v18; // edx
  __int32 *v19; // rdi
  int v20; // r14d
  __int64 v21; // r9
  unsigned int v22; // r8d
  __int64 v23; // rax
  __int32 v24; // ecx
  _DWORD *v25; // rdx
  _BYTE *v26; // rsi
  _QWORD *v27; // rdi
  int v28; // eax
  int v29; // eax
  __int64 v30; // rax
  char v31; // di
  int v32; // r11d
  int v33; // r11d
  __int64 v34; // r9
  __int64 v35; // rdx
  __int32 v36; // ecx
  int v37; // r8d
  __int32 *v38; // rsi
  int v39; // r11d
  int v40; // r11d
  __int64 v41; // r9
  __int64 v42; // rdx
  int v43; // r8d
  __int32 v44; // ecx
  __int64 v45; // [rsp+0h] [rbp-80h]
  unsigned __int8 v46; // [rsp+Fh] [rbp-71h]
  __int64 v47; // [rsp+10h] [rbp-70h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  _QWORD *v49; // [rsp+20h] [rbp-60h]
  __int32 v50; // [rsp+28h] [rbp-58h]
  unsigned __int64 v51; // [rsp+28h] [rbp-58h]
  __m128i v52; // [rsp+38h] [rbp-48h] BYREF

  sub_2DF8870(a1);
  *(_QWORD *)(a1 + 104) = a2;
  *(_QWORD *)(a1 + 120) = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v46 = sub_2DFE700(a1, a2, a3);
  sub_2E02A20(a1);
  v5 = *(_QWORD *)(a1 + 104);
  if ( *(_DWORD *)(v5 + 1096) )
  {
    v48 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32LL);
    v7 = *(_DWORD **)(v5 + 1088);
    v8 = &v7[6 * *(unsigned int *)(v5 + 1104)];
    if ( v7 != v8 )
    {
      while ( 1 )
      {
        v9 = v7;
        if ( *v7 <= 0xFFFFFFFD )
          break;
        v7 += 6;
        if ( v8 == v7 )
          goto LABEL_2;
      }
      if ( v8 != v7 )
      {
        v10 = *v7;
        v45 = a1 + 176;
        v49 = (_QWORD *)(a1 + 136);
        while ( 1 )
        {
          v11 = v9[4];
          v50 = v9[5];
          v52.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v48 + 152) + 16LL * *(unsigned int *)(*((_QWORD *)v9 + 1) + 24LL));
          v52.m128i_i32[3] = v50;
          v52.m128i_i32[2] = v11;
          v12 = sub_22077B0(0x38u);
          *(_DWORD *)(v12 + 32) = v10;
          v13 = v12;
          v52.m128i_i32[3] = v50;
          v14 = *(_QWORD *)(a1 + 144);
          *(__m128i *)(v12 + 40) = _mm_loadu_si128(&v52);
          if ( v14 )
          {
            while ( 1 )
            {
              v15 = *(_DWORD *)(v14 + 32);
              v16 = *(_QWORD *)(v14 + 24);
              if ( v15 > v10 )
                v16 = *(_QWORD *)(v14 + 16);
              if ( !v16 )
                break;
              v14 = v16;
            }
            if ( v10 >= v15 )
            {
              if ( v15 >= v10 )
                goto LABEL_17;
              goto LABEL_46;
            }
            if ( *(_QWORD *)(a1 + 152) == v14 )
              goto LABEL_46;
          }
          else
          {
            v14 = *(_QWORD *)(a1 + 152);
            if ( v49 == (_QWORD *)v14 )
            {
LABEL_47:
              v31 = 1;
              goto LABEL_48;
            }
            v14 = (__int64)v49;
          }
          v51 = v13;
          v47 = v14;
          v30 = sub_220EF80(v14);
          v13 = v51;
          if ( *(_DWORD *)(v30 + 32) >= v10 || (v14 = v47) == 0 )
          {
LABEL_17:
            j_j___libc_free_0(v13);
            v17 = *(_DWORD *)(a1 + 200);
            if ( !v17 )
              goto LABEL_49;
            goto LABEL_18;
          }
LABEL_46:
          if ( v49 == (_QWORD *)v14 )
            goto LABEL_47;
          v31 = *(_DWORD *)(v14 + 32) > v10;
LABEL_48:
          sub_220F040(v31, v13, (_QWORD *)v14, v49);
          v17 = *(_DWORD *)(a1 + 200);
          ++*(_QWORD *)(a1 + 168);
          if ( !v17 )
          {
LABEL_49:
            ++*(_QWORD *)(a1 + 176);
            goto LABEL_50;
          }
LABEL_18:
          v18 = 1;
          v19 = 0;
          v20 = 37 * v11;
          v21 = *(_QWORD *)(a1 + 184);
          v22 = (v17 - 1) & (37 * v11);
          v23 = v21 + 32LL * v22;
          v24 = *(_DWORD *)v23;
          if ( v11 != *(_DWORD *)v23 )
          {
            while ( v24 != -1 )
            {
              if ( !v19 && v24 == -2 )
                v19 = (__int32 *)v23;
              v22 = (v17 - 1) & (v18 + v22);
              v23 = v21 + 32LL * v22;
              v24 = *(_DWORD *)v23;
              if ( *(_DWORD *)v23 == v11 )
                goto LABEL_19;
              ++v18;
            }
            if ( !v19 )
              v19 = (__int32 *)v23;
            v28 = *(_DWORD *)(a1 + 192);
            ++*(_QWORD *)(a1 + 176);
            v29 = v28 + 1;
            if ( 4 * v29 >= 3 * v17 )
            {
LABEL_50:
              sub_2E00A10(v45, 2 * v17);
              v32 = *(_DWORD *)(a1 + 200);
              if ( !v32 )
                goto LABEL_73;
              v33 = v32 - 1;
              v34 = *(_QWORD *)(a1 + 184);
              LODWORD(v35) = v33 & (37 * v11);
              v29 = *(_DWORD *)(a1 + 192) + 1;
              v19 = (__int32 *)(v34 + 32LL * (unsigned int)v35);
              v36 = *v19;
              if ( v11 != *v19 )
              {
                v37 = 1;
                v38 = 0;
                while ( v36 != -1 )
                {
                  if ( !v38 && v36 == -2 )
                    v38 = v19;
                  v35 = v33 & (unsigned int)(v35 + v37);
                  v19 = (__int32 *)(v34 + 32 * v35);
                  v36 = *v19;
                  if ( v11 == *v19 )
                    goto LABEL_39;
                  ++v37;
                }
                goto LABEL_54;
              }
            }
            else if ( v17 - *(_DWORD *)(a1 + 196) - v29 <= v17 >> 3 )
            {
              sub_2E00A10(v45, v17);
              v39 = *(_DWORD *)(a1 + 200);
              if ( !v39 )
              {
LABEL_73:
                ++*(_DWORD *)(a1 + 192);
                BUG();
              }
              v40 = v39 - 1;
              v41 = *(_QWORD *)(a1 + 184);
              v38 = 0;
              LODWORD(v42) = v40 & v20;
              v43 = 1;
              v29 = *(_DWORD *)(a1 + 192) + 1;
              v19 = (__int32 *)(v41 + 32LL * (v40 & (unsigned int)v20));
              v44 = *v19;
              if ( v11 != *v19 )
              {
                while ( v44 != -1 )
                {
                  if ( v44 == -2 && !v38 )
                    v38 = v19;
                  v42 = v40 & (unsigned int)(v42 + v43);
                  v19 = (__int32 *)(v41 + 32 * v42);
                  v44 = *v19;
                  if ( v11 == *v19 )
                    goto LABEL_39;
                  ++v43;
                }
LABEL_54:
                if ( v38 )
                  v19 = v38;
              }
            }
LABEL_39:
            *(_DWORD *)(a1 + 192) = v29;
            if ( *v19 != -1 )
              --*(_DWORD *)(a1 + 196);
            *v19 = v11;
            v26 = 0;
            v27 = v19 + 2;
            *v27 = 0;
            v27[1] = 0;
            v27[2] = 0;
            goto LABEL_42;
          }
LABEL_19:
          v25 = *(_DWORD **)(v23 + 16);
          v26 = *(_BYTE **)(v23 + 24);
          v27 = (_QWORD *)(v23 + 8);
          if ( v25 != (_DWORD *)v26 )
          {
            if ( v25 )
            {
              *v25 = *v9;
              v25 = *(_DWORD **)(v23 + 16);
            }
            *(_QWORD *)(v23 + 16) = v25 + 1;
            goto LABEL_23;
          }
LABEL_42:
          sub_B8BBF0((__int64)v27, v26, v9);
LABEL_23:
          v9 += 6;
          if ( v9 == v8 )
            break;
          while ( *v9 > 0xFFFFFFFD )
          {
            v9 += 6;
            if ( v8 == v9 )
              goto LABEL_2;
          }
          if ( v8 == v9 )
            break;
          v10 = *v9;
        }
      }
    }
  }
LABEL_2:
  *(_BYTE *)(a1 + 993) = v46;
  return v46;
}
