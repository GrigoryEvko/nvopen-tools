// Function: sub_1194CD0
// Address: 0x1194cd0
//
__int64 __fastcall sub_1194CD0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r15d
  __int64 v6; // r14
  __int64 v8; // rbx
  unsigned __int8 v9; // r12
  __int64 v10; // rax
  char v12; // al
  unsigned int v13; // r12d
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rdi
  _QWORD *v19; // rdx
  unsigned __int32 v20; // eax
  __int64 v21; // r8
  unsigned int v22; // r15d
  unsigned __int64 v23; // rdx
  __int32 v24; // eax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 *v27; // rbx
  __int64 v28; // r9
  __int32 *v29; // rsi
  __int64 v30; // rcx
  __m128i *v31; // rdi
  __int64 v32; // rbx
  __int64 v33; // rdi
  unsigned int v34; // r14d
  __int64 v36; // rdx
  int v37; // edx
  unsigned __int64 v38; // rsi
  int v39; // esi
  unsigned int v41; // r15d
  __int64 v42; // r12
  __int64 v43; // rax
  _QWORD *v44; // r9
  unsigned int v45; // r15d
  _QWORD *v46; // r12
  _BYTE *v47; // rax
  int v48; // eax
  __int64 v49; // rdx
  _BYTE *v50; // rax
  int v51; // ebx
  __int64 v52; // [rsp+0h] [rbp-B0h]
  __int64 v53; // [rsp+10h] [rbp-A0h]
  __int64 v54; // [rsp+10h] [rbp-A0h]
  __int64 v55; // [rsp+10h] [rbp-A0h]
  unsigned int v56; // [rsp+18h] [rbp-98h]
  unsigned int v57; // [rsp+18h] [rbp-98h]
  unsigned __int64 v58; // [rsp+18h] [rbp-98h]
  __int64 v59; // [rsp+18h] [rbp-98h]
  _QWORD *v60; // [rsp+18h] [rbp-98h]
  char v61; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+18h] [rbp-98h]
  __int64 v63; // [rsp+18h] [rbp-98h]
  unsigned __int64 v64; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int32 v65; // [rsp+28h] [rbp-88h]
  __m128i v66[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v67; // [rsp+58h] [rbp-58h]
  unsigned int v68; // [rsp+A8h] [rbp-8h]

  while ( 2 )
  {
    v6 = a4;
    v8 = a1;
    v9 = *(_BYTE *)a1;
    if ( *(_BYTE *)a1 != 5 && v9 <= 0x15u )
    {
      v53 = a5;
      v56 = a3;
      v12 = sub_AD6CA0(a1);
      a3 = v56;
      a5 = v53;
      if ( !v12 )
        return 1;
      v9 = *(_BYTE *)a1;
    }
    if ( v9 <= 0x1Cu )
      return 0;
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v10 || *(_QWORD *)(v10 + 8) )
      return 0;
    switch ( v9 )
    {
      case '.':
        if ( (_BYTE)a3 )
          return 0;
        v32 = (*(_BYTE *)(a1 + 7) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v33 = *(_QWORD *)(v32 + 32);
        if ( *(_BYTE *)v33 != 17 )
        {
          v49 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v33 + 8) + 8LL) - 17;
          if ( (unsigned int)v49 > 1 )
            return 0;
          if ( *(_BYTE *)v33 > 0x15u )
            return 0;
          v50 = sub_AD7630(v33, 0, v49);
          v33 = (__int64)v50;
          if ( !v50 || *v50 != 17 )
            return 0;
        }
        v34 = *(_DWORD *)(v33 + 32);
        _RAX = *(_QWORD *)(v33 + 24);
        v36 = 1LL << ((unsigned __int8)v34 - 1);
        if ( v34 <= 0x40 )
        {
          if ( (v36 & _RAX) == 0 )
            return 0;
          if ( v34 )
          {
            v37 = 64;
            _BitScanReverse64(&v38, ~(_RAX << (64 - (unsigned __int8)v34)));
            v39 = v38 ^ 0x3F;
            if ( _RAX << (64 - (unsigned __int8)v34) != -1 )
              v37 = v39;
            __asm { tzcnt   rax, rax }
            if ( (unsigned int)_RAX > v34 )
              LODWORD(_RAX) = *(_DWORD *)(v33 + 32);
            if ( v34 == (_DWORD)_RAX + v37 )
              goto LABEL_66;
            return 0;
          }
          LODWORD(_RAX) = 0;
LABEL_66:
          LOBYTE(v5) = a2 == (_DWORD)_RAX;
          return v5;
        }
        if ( (*(_QWORD *)(_RAX + 8LL * ((v34 - 1) >> 6)) & v36) != 0 )
        {
          v51 = sub_C44500(v33 + 24);
          LODWORD(_RAX) = sub_C44590(v33 + 24);
          if ( (_DWORD)_RAX + v51 == v34 )
            goto LABEL_66;
        }
        return 0;
      case '6':
      case '7':
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v16 = *(_QWORD *)(a1 - 8);
        else
          v16 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v17 = *(_QWORD *)(v16 + 32);
        if ( *(_BYTE *)v17 == 17 )
        {
          v18 = v17 + 24;
        }
        else
        {
          v55 = a5;
          v61 = a3;
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v17 + 8) + 8LL) - 17 > 1 )
            return 0;
          if ( *(_BYTE *)v17 > 0x15u )
            return 0;
          v47 = sub_AD7630(v17, 0, a3);
          if ( !v47 || *v47 != 17 )
            return 0;
          v9 = *(_BYTE *)v8;
          LOBYTE(a3) = v61;
          v18 = (__int64)(v47 + 24);
          a5 = v55;
        }
        v5 = 1;
        if ( (_BYTE)a3 == (v9 == 54) )
          return v5;
        v57 = *(_DWORD *)(v18 + 8);
        if ( v57 <= 0x40 )
        {
          v19 = *(_QWORD **)v18;
          if ( a2 == *(_QWORD *)v18 )
            return v5;
          if ( (unsigned __int64)a2 >= *(_QWORD *)v18 )
            return 0;
LABEL_27:
          v54 = a5;
          v58 = (unsigned __int64)v19;
          v20 = sub_BCB060(*(_QWORD *)(v8 + 8));
          v21 = v54;
          if ( v58 >= v20 )
            return 0;
          v22 = v58 - a2;
          if ( v9 == 54 )
            v22 = v20 - v58;
          v66[0].m128i_i32[2] = v20;
          if ( v20 > 0x40 )
          {
            sub_C43690((__int64)v66, 0, 0);
            v21 = v54;
          }
          else
          {
            v66[0].m128i_i64[0] = 0;
          }
          if ( a2 )
          {
            if ( a2 > 0x40 )
            {
              v63 = v21;
              sub_C43C90(v66, 0, a2);
              v21 = v63;
            }
            else
            {
              v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2);
              if ( v66[0].m128i_i32[2] > 0x40u )
                *(_QWORD *)v66[0].m128i_i64[0] |= v23;
              else
                v66[0].m128i_i64[0] |= v23;
            }
          }
          v24 = v66[0].m128i_i32[2];
          v65 = v66[0].m128i_u32[2];
          if ( v66[0].m128i_i32[2] > 0x40u )
          {
            v62 = v21;
            sub_C43780((__int64)&v64, (const void **)v66);
            v24 = v65;
            v21 = v62;
            if ( v65 > 0x40 )
            {
              sub_C47690((__int64 *)&v64, v22);
              v21 = v62;
LABEL_43:
              if ( v66[0].m128i_i32[2] > 0x40u && v66[0].m128i_i64[0] )
              {
                v59 = v21;
                j_j___libc_free_0_0(v66[0].m128i_i64[0]);
                v21 = v59;
              }
              if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
                v27 = *(__int64 **)(v8 - 8);
              else
                v27 = (__int64 *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
              v28 = *v27;
              v29 = (__int32 *)(v6 + 96);
              v30 = 18;
              v31 = v66;
              while ( v30 )
              {
                v31->m128i_i32[0] = *v29++;
                v31 = (__m128i *)((char *)v31 + 4);
                --v30;
              }
              v67 = v21;
              v5 = sub_9AC230(v28, (__int64)&v64, v66, 0);
              if ( v65 > 0x40 && v64 )
                j_j___libc_free_0_0(v64);
              return v5;
            }
          }
          else
          {
            v64 = v66[0].m128i_i64[0];
          }
          if ( v24 == v22 )
            v25 = 0;
          else
            v25 = v64 << v22;
          v26 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v24;
          if ( !v24 )
            v26 = 0;
          v64 = v26 & v25;
          goto LABEL_43;
        }
        v52 = a5;
        v48 = sub_C444A0(v18);
        a5 = v52;
        if ( v57 - v48 > 0x40 )
          return 0;
        v19 = **(_QWORD ***)v18;
        if ( (_QWORD *)a2 != v19 )
        {
          if ( a2 >= (unsigned __int64)v19 )
            return 0;
          goto LABEL_27;
        }
        return v5;
      case '9':
      case ':':
      case ';':
        v13 = (unsigned __int8)a3;
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v14 = *(_QWORD **)(a1 - 8);
        else
          v14 = (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
        if ( !(unsigned __int8)sub_1194CD0(*v14, a2, (unsigned __int8)a3, v6, v8) )
          return 0;
        if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
          v15 = *(_QWORD *)(v8 - 8);
        else
          v15 = v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
        a1 = *(_QWORD *)(v15 + 32);
        a5 = v8;
        a4 = v6;
        a3 = v13;
        goto LABEL_18;
      case 'T':
        v43 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        {
          v44 = *(_QWORD **)(a1 - 8);
          v60 = &v44[v43];
        }
        else
        {
          v60 = (_QWORD *)a1;
          v44 = (_QWORD *)(a1 - v43 * 8);
        }
        if ( v44 == v60 )
          return 1;
        v45 = (unsigned __int8)a3;
        v46 = v44;
        while ( (unsigned __int8)sub_1194CD0(*v46, a2, v45, v6, a1) )
        {
          v46 += 4;
          if ( v60 == v46 )
            return 1;
        }
        return 0;
      case 'V':
        v41 = (unsigned __int8)a3;
        v42 = *(_QWORD *)(a1 - 32);
        if ( !(unsigned __int8)sub_1194CD0(*(_QWORD *)(a1 - 64), a2, (unsigned __int8)a3, v6, a1) )
          return 0;
        a5 = a1;
        a4 = v6;
        a3 = v41;
        a1 = v42;
LABEL_18:
        v5 = v68;
        continue;
      default:
        return 0;
    }
  }
}
