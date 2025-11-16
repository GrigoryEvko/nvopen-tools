// Function: sub_1F746A0
// Address: 0x1f746a0
//
__int64 __fastcall sub_1F746A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v9; // rbx
  unsigned int *v11; // rax
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rax
  __int16 v17; // r15
  __int64 v19; // rdx
  char v20; // r9
  unsigned int v21; // eax
  __int8 v22; // r9
  __int64 v23; // rdx
  unsigned int v24; // eax
  unsigned int v25; // r15d
  unsigned int v26; // eax
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rsi
  unsigned int v30; // edx
  __int64 v31; // r15
  __int64 v32; // rcx
  __int64 v33; // r15
  bool v34; // al
  __int64 *v35; // rax
  unsigned int v36; // eax
  unsigned int v37; // eax
  _BYTE *v38; // rdx
  bool v39; // di
  __int64 v40; // rsi
  __int64 v41; // rdi
  bool v44; // cc
  char v45; // di
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  char v49; // r15
  __int64 v50; // rax
  unsigned __int32 v51; // edx
  __int64 *v52; // rsi
  unsigned int v53; // edi
  __int64 *v54; // rcx
  char v55; // al
  __int64 v56; // rax
  unsigned int v57; // eax
  __int64 *v58; // rsi
  unsigned int v59; // edi
  __int64 *v60; // rcx
  __int64 v61; // [rsp+0h] [rbp-A0h]
  __int64 v62; // [rsp+0h] [rbp-A0h]
  __int64 v63; // [rsp+8h] [rbp-98h]
  __int64 v64; // [rsp+8h] [rbp-98h]
  __int8 v65; // [rsp+8h] [rbp-98h]
  char v66; // [rsp+8h] [rbp-98h]
  unsigned int v67; // [rsp+10h] [rbp-90h]
  bool v68; // [rsp+10h] [rbp-90h]
  unsigned int v69; // [rsp+10h] [rbp-90h]
  __int64 v70; // [rsp+10h] [rbp-90h]
  bool v71; // [rsp+10h] [rbp-90h]
  __int64 v76; // [rsp+40h] [rbp-60h]
  __m128i v78; // [rsp+50h] [rbp-50h] BYREF
  __int64 v79; // [rsp+60h] [rbp-40h] BYREF
  __int64 v80; // [rsp+68h] [rbp-38h]

  v7 = *(unsigned int *)(a2 + 56);
  if ( !(_DWORD)v7 )
    return 1;
  v9 = 0;
  v76 = 40 * v7;
  while ( 1 )
  {
    v11 = (unsigned int *)(v9 + *(_QWORD *)(a2 + 32));
    v12 = *(_QWORD *)v11;
    v13 = v11[2];
    v14 = *(_QWORD *)(*(_QWORD *)v11 + 40LL) + 16 * v13;
    v15 = *(_BYTE *)v14;
    v16 = *(_QWORD *)(v14 + 8);
    LOBYTE(v79) = v15;
    v80 = v16;
    if ( v15 ? (unsigned __int8)(v15 - 14) <= 0x5Fu : sub_1F58D20((__int64)&v79) )
      return 0;
    LOBYTE(v6) = *(_WORD *)(v12 + 24) == 32 || *(_WORD *)(v12 + 24) == 10;
    if ( (_BYTE)v6 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a2 + 24) - 119 > 1 )
        goto LABEL_7;
      v28 = *(_QWORD *)(v12 + 88);
      v29 = *(_QWORD *)(a5 + 88);
      v30 = *(_DWORD *)(v29 + 32);
      v78.m128i_i32[2] = v30;
      if ( v30 <= 0x40 )
      {
        v31 = *(_QWORD *)(v29 + 24);
LABEL_36:
        v32 = *(_QWORD *)(v28 + 24);
        LODWORD(v80) = v30;
        v78.m128i_i32[2] = 0;
        v33 = v32 & v31;
        v78.m128i_i64[0] = v33;
        v79 = v33;
        goto LABEL_37;
      }
      v70 = v28;
      sub_16A4FD0((__int64)&v78, (const void **)(v29 + 24));
      v30 = v78.m128i_u32[2];
      v28 = v70;
      if ( v78.m128i_i32[2] <= 0x40u )
      {
        v31 = v78.m128i_i64[0];
        goto LABEL_36;
      }
      sub_16A8890(v78.m128i_i64, (__int64 *)(v70 + 24));
      v51 = v78.m128i_u32[2];
      v33 = v78.m128i_i64[0];
      v78.m128i_i32[2] = 0;
      LODWORD(v80) = v51;
      v79 = v78.m128i_i64[0];
      if ( v51 > 0x40 )
      {
        v34 = !sub_16A5220((__int64)&v79, (const void **)(v70 + 24));
        if ( v33 )
        {
          v71 = v34;
          j_j___libc_free_0_0(v33);
          v34 = v71;
        }
        if ( v78.m128i_i32[2] > 0x40u && v78.m128i_i64[0] )
        {
          v68 = v34;
          j_j___libc_free_0_0(v78.m128i_i64[0]);
          v34 = v68;
        }
        if ( !v34 )
          goto LABEL_7;
        goto LABEL_41;
      }
      v32 = *(_QWORD *)(v70 + 24);
LABEL_37:
      if ( v33 == v32 )
        goto LABEL_7;
LABEL_41:
      v35 = *(__int64 **)(a4 + 8);
      if ( *(__int64 **)(a4 + 16) == v35 )
      {
        v52 = &v35[*(unsigned int *)(a4 + 28)];
        v53 = *(_DWORD *)(a4 + 28);
        if ( v35 != v52 )
        {
          v54 = 0;
          while ( a2 != *v35 )
          {
            if ( *v35 == -2 )
              v54 = v35;
            if ( v52 == ++v35 )
            {
              if ( !v54 )
                goto LABEL_103;
              *v54 = a2;
              --*(_DWORD *)(a4 + 32);
              ++*(_QWORD *)a4;
              goto LABEL_7;
            }
          }
          goto LABEL_7;
        }
LABEL_103:
        if ( v53 < *(_DWORD *)(a4 + 24) )
        {
          *(_DWORD *)(a4 + 28) = v53 + 1;
          *v52 = a2;
          ++*(_QWORD *)a4;
          goto LABEL_7;
        }
      }
      v9 += 40;
      sub_16CCBA0(a4, a2);
      if ( v76 == v9 )
        return 1;
    }
    else
    {
      if ( !sub_1D18C00(v12, 1, v13) )
        return 0;
      v17 = *(_WORD *)(v12 + 24);
      if ( v17 == 143 )
        goto LABEL_55;
      if ( v17 <= 143 )
      {
        if ( v17 != 4 )
        {
          if ( (unsigned __int16)(v17 - 118) <= 2u )
          {
            if ( !(unsigned __int8)sub_1F746A0(a1, v12, a3, a4, a5, a6) )
              return 0;
            goto LABEL_7;
          }
          goto LABEL_48;
        }
LABEL_55:
        v41 = *(_QWORD *)(a5 + 88);
        if ( *(_DWORD *)(v41 + 32) <= 0x40u )
        {
          _RSI = ~*(_QWORD *)(v41 + 24);
          if ( *(_QWORD *)(v41 + 24) != -1 )
          {
            __asm { tzcnt   rsi, rsi }
            v44 = (unsigned int)_RSI <= 0x20;
            if ( (_DWORD)_RSI != 32 )
              goto LABEL_58;
LABEL_85:
            v45 = 5;
            goto LABEL_61;
          }
LABEL_95:
          v45 = 6;
          goto LABEL_61;
        }
        LODWORD(_RSI) = sub_16A58F0(v41 + 24);
        v44 = (unsigned int)_RSI <= 0x20;
        if ( (_DWORD)_RSI == 32 )
          goto LABEL_85;
LABEL_58:
        if ( v44 )
        {
          if ( (_DWORD)_RSI == 8 )
          {
            v45 = 3;
            goto LABEL_61;
          }
          v45 = 4;
          if ( (_DWORD)_RSI != 16 )
          {
            v45 = 2;
            if ( (_DWORD)_RSI != 1 )
              goto LABEL_91;
          }
LABEL_61:
          v46 = 0;
        }
        else
        {
          if ( (_DWORD)_RSI == 64 )
            goto LABEL_95;
          if ( (_DWORD)_RSI == 128 )
          {
            v45 = 7;
            goto LABEL_61;
          }
LABEL_91:
          v55 = sub_1F58CC0(*(_QWORD **)(*(_QWORD *)a1 + 48LL), _RSI);
          v17 = *(_WORD *)(v12 + 24);
          v45 = v55;
        }
        v78.m128i_i8[0] = v45;
        v78.m128i_i64[1] = v46;
        v47 = *(_QWORD *)(v12 + 32);
        if ( v17 == 4 )
        {
          v56 = *(_QWORD *)(v47 + 40);
          v49 = *(_BYTE *)(v56 + 88);
          v50 = *(_QWORD *)(v56 + 96);
        }
        else
        {
          v48 = *(_QWORD *)(*(_QWORD *)v47 + 40LL) + 16LL * *(unsigned int *)(v47 + 8);
          v49 = *(_BYTE *)v48;
          v50 = *(_QWORD *)(v48 + 8);
        }
        LOBYTE(v79) = v49;
        v80 = v50;
        if ( v49 == v45 )
        {
          if ( v49 || v50 == v46 )
            goto LABEL_7;
LABEL_67:
          v69 = sub_1F58D40((__int64)&v78);
          if ( !v49 )
            goto LABEL_68;
LABEL_46:
          v36 = sub_1F6C8D0(v49);
        }
        else
        {
          if ( !v45 )
            goto LABEL_67;
          v69 = sub_1F6C8D0(v45);
          if ( v49 )
            goto LABEL_46;
LABEL_68:
          v36 = sub_1F58D40((__int64)&v79);
        }
        if ( v36 <= v69 )
          goto LABEL_7;
LABEL_48:
        if ( *a6 )
          return 0;
        *a6 = v12;
        v37 = *(_DWORD *)(v12 + 60);
        if ( v37 > 1 )
        {
          v38 = *(_BYTE **)(v12 + 40);
          v39 = 0;
          v40 = (__int64)&v38[16 * v37];
          while ( 1 )
          {
            if ( *v38 != 1 && *v38 != 111 )
            {
              if ( v39 )
              {
                *a6 = 0;
                return v6;
              }
              v39 = *v38 != 1 && *v38 != 111;
            }
            v38 += 16;
            if ( (_BYTE *)v40 == v38 )
              goto LABEL_7;
          }
        }
        goto LABEL_7;
      }
      if ( v17 != 185 )
        goto LABEL_48;
      v78.m128i_i8[0] = 0;
      v78.m128i_i64[1] = 0;
      if ( !(unsigned __int8)sub_1F70D60(a1, *(_QWORD *)(a5 + 88), v12, **(_BYTE **)(v12 + 40), &v78)
        || !(unsigned __int8)sub_1F742D0(a1, v12, 3u, (unsigned int *)&v78, 0) )
      {
        return v6;
      }
      v6 = *(unsigned __int8 *)(v12 + 88);
      v19 = *(_QWORD *)(v12 + 96);
      v20 = v78.m128i_i8[0];
      if ( ((*(_BYTE *)(v12 + 27) ^ 0xC) & 0xC) == 0 )
      {
        LOBYTE(v79) = *(_BYTE *)(v12 + 88);
        v80 = v19;
        if ( v78.m128i_i8[0] == (_BYTE)v6 )
        {
          if ( v78.m128i_i8[0] || v78.m128i_i64[1] == v19 )
            goto LABEL_7;
        }
        else if ( v78.m128i_i8[0] )
        {
          v63 = v19;
          v21 = sub_1F6C8D0(v78.m128i_i8[0]);
          v23 = v63;
          v67 = v21;
          if ( (_BYTE)v6 )
            goto LABEL_25;
LABEL_100:
          v62 = v23;
          v66 = v22;
          v24 = sub_1F58D40((__int64)&v79);
          v19 = v62;
          v20 = v66;
          goto LABEL_26;
        }
        v61 = v19;
        v65 = v78.m128i_i8[0];
        v57 = sub_1F58D40((__int64)&v78);
        v23 = v61;
        v22 = v65;
        v67 = v57;
        if ( !(_BYTE)v6 )
          goto LABEL_100;
LABEL_25:
        v64 = v23;
        v24 = sub_1F6C8D0(v6);
        v19 = v64;
LABEL_26:
        if ( v24 <= v67 )
          goto LABEL_7;
      }
      LOBYTE(v79) = v6;
      v80 = v19;
      if ( v20 != (_BYTE)v6 )
      {
        if ( v20 )
        {
          v25 = sub_1F6C8D0(v20);
          if ( (_BYTE)v6 )
            goto LABEL_30;
LABEL_89:
          v26 = sub_1F58D40((__int64)&v79);
        }
        else
        {
LABEL_88:
          v25 = sub_1F58D40((__int64)&v78);
          if ( !(_BYTE)v6 )
            goto LABEL_89;
LABEL_30:
          v26 = sub_1F6C8D0(v6);
        }
        if ( v26 < v25 )
          goto LABEL_7;
        goto LABEL_32;
      }
      if ( !(_BYTE)v6 && v78.m128i_i64[1] != v19 )
        goto LABEL_88;
LABEL_32:
      v27 = *(__int64 **)(a3 + 8);
      if ( *(__int64 **)(a3 + 16) != v27 )
        goto LABEL_33;
      v58 = &v27[*(unsigned int *)(a3 + 28)];
      v59 = *(_DWORD *)(a3 + 28);
      if ( v27 != v58 )
      {
        v60 = 0;
        while ( v12 != *v27 )
        {
          if ( *v27 == -2 )
            v60 = v27;
          if ( v58 == ++v27 )
          {
            if ( !v60 )
              goto LABEL_113;
            *v60 = v12;
            --*(_DWORD *)(a3 + 32);
            ++*(_QWORD *)a3;
            goto LABEL_7;
          }
        }
        goto LABEL_7;
      }
LABEL_113:
      if ( v59 < *(_DWORD *)(a3 + 24) )
      {
        *(_DWORD *)(a3 + 28) = v59 + 1;
        *v58 = v12;
        ++*(_QWORD *)a3;
      }
      else
      {
LABEL_33:
        sub_16CCBA0(a3, v12);
      }
LABEL_7:
      v9 += 40;
      if ( v76 == v9 )
        return 1;
    }
  }
}
