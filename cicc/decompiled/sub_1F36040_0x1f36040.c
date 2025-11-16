// Function: sub_1F36040
// Address: 0x1f36040
//
void __fastcall sub_1F36040(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r15
  int v14; // eax
  int v15; // eax
  __int64 v16; // rsi
  int v17; // edi
  unsigned int v18; // edx
  int v19; // ecx
  __int64 v20; // r12
  int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned int v24; // edx
  _DWORD *v25; // r10
  int v26; // esi
  __int64 *v27; // rdi
  int v28; // esi
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r8
  unsigned __int64 v32; // r11
  __int64 v33; // r10
  __int64 v34; // r9
  int v35; // esi
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // r8
  int v39; // r9d
  __int64 v40; // rsi
  __int64 v41; // r8
  __int64 v42; // r9
  _DWORD *v43; // r10
  int v44; // eax
  __int64 v46; // r13
  __int64 v47; // r14
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  int v52; // r10d
  __int64 v53; // [rsp+0h] [rbp-E0h]
  __int64 v54; // [rsp+8h] [rbp-D8h]
  __int64 v55; // [rsp+10h] [rbp-D0h]
  __m128i *v56; // [rsp+18h] [rbp-C8h]
  __int64 *v57; // [rsp+20h] [rbp-C0h]
  _DWORD *v58; // [rsp+28h] [rbp-B8h]
  __int64 v59; // [rsp+28h] [rbp-B8h]
  __int64 v60; // [rsp+28h] [rbp-B8h]
  _DWORD *v61; // [rsp+30h] [rbp-B0h]
  __int64 v62; // [rsp+30h] [rbp-B0h]
  _QWORD *v63; // [rsp+30h] [rbp-B0h]
  __int64 v64; // [rsp+38h] [rbp-A8h]
  int v65; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v66; // [rsp+40h] [rbp-A0h]
  __int64 v67; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v68; // [rsp+40h] [rbp-A0h]
  __int64 v69; // [rsp+40h] [rbp-A0h]
  int v70; // [rsp+40h] [rbp-A0h]
  __int64 v75; // [rsp+68h] [rbp-78h]
  __int64 v76; // [rsp+70h] [rbp-70h] BYREF
  int v77; // [rsp+78h] [rbp-68h]
  __m128i v78; // [rsp+80h] [rbp-60h] BYREF
  __int64 v79; // [rsp+90h] [rbp-50h]
  __int64 v80; // [rsp+98h] [rbp-48h]
  __int64 v81; // [rsp+A0h] [rbp-40h]

  v7 = a4 + 3;
  v8 = *(_QWORD **)a1;
  v64 = a6;
  if ( **(_WORD **)(a2 + 16) == 2 )
  {
    v46 = v8[1];
    sub_1DD6ED0(&v76, (__int64)a4, a4[4]);
    v47 = a4[7];
    v48 = (__int64)sub_1E0B640(v47, v46 + 128, &v76, 0);
    sub_1DD5BA0(a4 + 2, v48);
    v49 = a4[3];
    *(_QWORD *)(v48 + 8) = v7;
    *(_QWORD *)v48 = v49 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v48 & 7LL;
    *(_QWORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v48;
    a4[3] = v48 | a4[3] & 7;
    LODWORD(v49) = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
    v78.m128i_i64[0] = 16;
    v79 = 0;
    LODWORD(v80) = v49;
    sub_1E1A9C0(v48, v47, &v78);
    if ( v76 )
      sub_161E7C0((__int64)&v76, v76);
  }
  else
  {
    v9 = (*(__int64 (__fastcall **)(_QWORD *, __int64 *, __int64 *, __int64))(*v8 + 160LL))(v8, a4, a4 + 3, a2);
    v10 = v9;
    if ( *(_BYTE *)(a1 + 48) )
    {
      v11 = *(unsigned int *)(v9 + 40);
      if ( (_DWORD)v11 )
      {
        v75 = v11;
        v12 = 0;
        v13 = v10;
        v57 = (__int64 *)(a2 + 64);
        do
        {
          while ( 1 )
          {
            v20 = *(_QWORD *)(v13 + 32) + 40 * v12;
            if ( *(_BYTE *)v20 )
              goto LABEL_9;
            v21 = *(_DWORD *)(v20 + 8);
            if ( v21 >= 0 )
              goto LABEL_9;
            if ( (*(_BYTE *)(v20 + 3) & 0x10) == 0 )
              break;
            v65 = sub_1E6B9A0(
                    *(_QWORD *)(a1 + 32),
                    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 24LL) + 16LL * (v21 & 0x7FFFFFFF))
                  & 0xFFFFFFFFFFFFFFF8LL,
                    (unsigned __int8 *)byte_3F871B3,
                    0,
                    v10,
                    v12);
            sub_1E310D0(v20, v65);
            v76 = __PAIR64__(v65, v21);
            v77 = 0;
            sub_1F35DB0((__int64)&v78, a5, (int *)&v76, (__int64 *)((char *)&v76 + 4));
            if ( (unsigned __int8)sub_1F33B40(v21, a3, *(_QWORD *)(a1 + 32)) )
              goto LABEL_8;
            v14 = *(_DWORD *)(v64 + 24);
            if ( v14 )
            {
              v15 = v14 - 1;
              v16 = *(_QWORD *)(v64 + 8);
              v17 = 1;
              v18 = v15 & (37 * v21);
              v19 = *(_DWORD *)(v16 + 4LL * v18);
              if ( v21 != v19 )
              {
                while ( v19 != -1 )
                {
                  v10 = (unsigned int)(v17 + 1);
                  v18 = v15 & (v17 + v18);
                  v19 = *(_DWORD *)(v16 + 4LL * v18);
                  if ( v21 == v19 )
                    goto LABEL_8;
                  ++v17;
                }
                goto LABEL_9;
              }
LABEL_8:
              sub_1F357A0(a1, v21, v65, (__int64)a4);
            }
LABEL_9:
            if ( ++v12 == v75 )
              return;
          }
          v22 = *(unsigned int *)(a5 + 24);
          if ( !(_DWORD)v22 )
            goto LABEL_9;
          v23 = *(_QWORD *)(a5 + 8);
          v24 = (v22 - 1) & (37 * v21);
          v25 = (_DWORD *)(v23 + 12LL * v24);
          v26 = *v25;
          if ( v21 != *v25 )
          {
            v52 = 1;
            while ( v26 != -1 )
            {
              v10 = (unsigned int)(v52 + 1);
              v24 = (v22 - 1) & (v52 + v24);
              v25 = (_DWORD *)(v23 + 12LL * v24);
              v26 = *v25;
              if ( v21 == *v25 )
                goto LABEL_15;
              v52 = v10;
            }
            goto LABEL_9;
          }
LABEL_15:
          if ( v25 == (_DWORD *)(v23 + 12 * v22) )
            goto LABEL_9;
          v27 = *(__int64 **)(a1 + 32);
          LODWORD(v58) = v12;
          v61 = v25;
          v28 = v25[1];
          v29 = v27[3];
          if ( v25[2] )
          {
            v66 = *(_QWORD *)(v29 + 16LL * (v21 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
            v30 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, unsigned __int64))(**(_QWORD **)(a1 + 8) + 96LL))(
                    *(_QWORD *)(a1 + 8),
                    *(_QWORD *)(v29 + 16LL * (v28 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                    v66);
            v32 = v66;
            v33 = (__int64)v61;
            v34 = (unsigned int)v12;
            if ( !v30 )
              goto LABEL_26;
            sub_1E693D0(*(_QWORD *)(a1 + 32), v61[1], v30);
            v33 = (__int64)v61;
          }
          else
          {
            v68 = *(_QWORD *)(v29 + 16LL * (v21 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
            v36 = sub_1E69410(v27, v28, v68, 0);
            v32 = v68;
            v33 = (__int64)v61;
            v34 = (unsigned int)v12;
            if ( !v36 )
            {
LABEL_26:
              v69 = v32;
              v37 = sub_1E16DA0(
                      a2,
                      v34,
                      *(_QWORD *)a1,
                      *(_QWORD **)(a1 + 8),
                      v31,
                      v34,
                      v53,
                      v54,
                      v55,
                      (__int64)v56,
                      (__int64)v57,
                      (__int64)v58,
                      v33,
                      v64);
              if ( !v37 )
                v37 = v69;
              v70 = sub_1E6B9A0(*(_QWORD *)(a1 + 32), v37, (unsigned __int8 *)byte_3F871B3, 0, v38, v39);
              v40 = *(_QWORD *)(*(_QWORD *)a1 + 8LL) + 960LL;
              if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
              {
                v55 = v62;
                v59 = a4[7];
                v63 = sub_1E0B640(v59, v40, v57, 0);
                sub_1DD6E10((__int64)a4, (__int64 *)a2, (__int64)v63);
                v78.m128i_i64[0] = 0x10000000;
                v56 = &v78;
                v79 = 0;
                v78.m128i_i32[2] = v70;
                v80 = 0;
                v81 = 0;
                sub_1E1A9C0((__int64)v63, v59, &v78);
                v41 = (__int64)v63;
                v42 = v59;
                v43 = (_DWORD *)v55;
              }
              else
              {
                v54 = v62;
                v55 = a4[7];
                v60 = (__int64)sub_1E0B640(v55, v40, v57, 0);
                sub_1DD5BA0(a4 + 2, v60);
                v56 = &v78;
                v50 = *(_QWORD *)a2;
                *(_QWORD *)(v60 + 8) = a2;
                *(_QWORD *)v60 = v50 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v60 & 7LL;
                *(_QWORD *)((v50 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v60;
                v51 = *(_QWORD *)a2;
                v78.m128i_i32[1] = 0;
                v79 = 0;
                v80 = 0;
                *(_QWORD *)a2 = v60 | v51 & 7;
                v81 = 0;
                v78.m128i_i32[2] = v70;
                v78.m128i_i32[0] = 0x10000000;
                sub_1E1A9C0(v60, v55, &v78);
                v43 = (_DWORD *)v62;
                v42 = v55;
                v41 = v60;
              }
              v58 = v43;
              v44 = v43[2] & 0xFFF;
              v78.m128i_i32[2] = v43[1];
              v78.m128i_i64[0] = (unsigned int)(v44 << 8);
              v79 = 0;
              v80 = 0;
              v81 = 0;
              sub_1E1A9C0(v41, v42, &v78);
              *v58 = -2;
              --*(_DWORD *)(a5 + 16);
              ++*(_DWORD *)(a5 + 20);
              v76 = __PAIR64__(v70, v21);
              v77 = 0;
              sub_1F35DB0((__int64)&v78, a5, (int *)&v76, (__int64 *)((char *)&v76 + 4));
              sub_1E310D0(v20, v70);
              goto LABEL_23;
            }
          }
          v67 = v33;
          sub_1E310D0(v20, *(_DWORD *)(v33 + 4));
          v35 = (*(_DWORD *)v20 >> 8) & 0xFFF;
          if ( v35 )
          {
            if ( *(_DWORD *)(v67 + 8) )
              v35 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 120LL))(*(_QWORD *)(a1 + 8)) & 0xFFF;
          }
          else
          {
            v35 = *(_DWORD *)(v67 + 8) & 0xFFF;
          }
          *(_DWORD *)v20 = *(_DWORD *)v20 & 0xFFF000FF | (v35 << 8);
LABEL_23:
          *(_BYTE *)(v20 + 3) &= ~0x40u;
          ++v12;
        }
        while ( v12 != v75 );
      }
    }
  }
}
