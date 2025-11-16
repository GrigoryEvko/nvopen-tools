// Function: sub_1E36300
// Address: 0x1e36300
//
unsigned __int64 __fastcall sub_1E36300(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int32 v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  __int32 v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int32 v17; // eax
  __int32 v18; // eax
  __int32 v19; // eax
  char v20; // al
  __int64 v21; // r14
  int v22; // ebx
  __int8 *v23; // rax
  __int8 *v24; // rax
  __int64 v25; // rbx
  char *v26; // r15
  char *v27; // r14
  __m128i v28; // xmm0
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // r10
  unsigned __int64 v34; // rdx
  __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  __int64 v37; // r11
  unsigned __int64 v38; // r9
  unsigned __int64 v39; // rsi
  __m128i v40; // [rsp+10h] [rbp-110h] BYREF
  __m128i v41; // [rsp+20h] [rbp-100h] BYREF
  __m128i v42; // [rsp+30h] [rbp-F0h] BYREF
  unsigned __int64 v43; // [rsp+40h] [rbp-E0h]
  __int64 v44; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v45; // [rsp+60h] [rbp-C0h] BYREF
  __int64 src; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+80h] [rbp-A0h]
  __int64 v49; // [rsp+88h] [rbp-98h]
  __int64 v50; // [rsp+90h] [rbp-90h]
  __int64 v51; // [rsp+98h] [rbp-88h]
  __int64 v52; // [rsp+A0h] [rbp-80h]
  __int64 v53; // [rsp+A8h] [rbp-78h]
  __m128i v54; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v55; // [rsp+C0h] [rbp-60h]
  __m128i v56; // [rsp+D0h] [rbp-50h]
  unsigned __int64 v57; // [rsp+E0h] [rbp-40h]
  unsigned __int64 v58; // [rsp+E8h] [rbp-38h]

  switch ( *(_BYTE *)a1 )
  {
    case 0:
      v20 = *(_BYTE *)(a1 + 3);
      LOBYTE(v44) = 0;
      LOBYTE(v45) = (v20 & 0x10) != 0;
      dest.m128i_i32[0] = (*(_DWORD *)a1 >> 8) & 0xFFF;
      LODWORD(src) = *(_DWORD *)(a1 + 8);
      result = sub_1E35590((char *)&v44, (int *)&src, dest.m128i_i32, (char *)&v45);
      break;
    case 1:
      v21 = *(_QWORD *)(a1 + 24);
      v22 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
      {
        v32 = unk_4FA04C8;
        if ( !unk_4FA04C8 )
          v32 = 0xFF51AFD7ED558CCDLL;
        qword_4F99938 = v32;
        sub_2207640(byte_4F99930);
      }
      v58 = qword_4F99938;
      v44 = 0;
      v23 = sub_1E360F0(&dest, &v44, dest.m128i_i8, (unsigned __int64)&v54, 1);
      v45 = v44;
      v24 = sub_15B2130(&dest, &v45, v23, (unsigned __int64)&v54, v22);
      src = v21;
      v25 = v45;
      v26 = v24 + 8;
      if ( v24 + 8 <= (__int8 *)&v54 )
      {
        *(_QWORD *)v24 = v21;
      }
      else
      {
        v27 = (char *)((char *)&v54 - v24);
        memcpy(v24, &src, (char *)&v54 - v24);
        if ( v25 )
        {
          v33 = dest.m128i_i64[0] - 0x4B6D499041670D8DLL * v56.m128i_i64[0];
          v34 = v57
              ^ (0xB492B66FBE98F273LL
               * __ROL8__(dest.m128i_i64[1] + v55.m128i_i64[1] + v54.m128i_i64[1] + v54.m128i_i64[0], 27));
          v54.m128i_i64[1] = v51
                           + v55.m128i_i64[1]
                           - 0x4B6D499041670D8DLL * __ROL8__(v52 + v56.m128i_i64[0] + v54.m128i_i64[1], 22);
          v35 = v56.m128i_i64[1] + v55.m128i_i64[0];
          v55.m128i_i64[0] = v34;
          v36 = 0xB492B66FBE98F273LL * __ROL8__(v35, 31);
          v37 = v33 + v48 + dest.m128i_i64[1];
          v55.m128i_i64[1] = v49 + v37;
          v38 = v50 + v57 + v36;
          v54.m128i_i64[0] = v36;
          v56.m128i_i64[0] = __ROL8__(v37, 20) + v33 + __ROR8__(v34 + v33 + v49 + v56.m128i_i64[1], 21);
          v39 = v38 + v51 + v52;
          v25 += 64;
          v56.m128i_i64[1] = v53 + v39;
          v57 = __ROL8__(v39, 20) + v38 + __ROR8__(v38 + v48 + v53 + v54.m128i_i64[1], 21);
        }
        else
        {
          v25 = 64;
          sub_15938B0((unsigned __int64 *)&v40, dest.m128i_i64, v58);
          v28 = _mm_loadu_si128(&v40);
          v29 = _mm_loadu_si128(&v41);
          v30 = _mm_loadu_si128(&v42);
          v57 = v43;
          v54 = v28;
          v55 = v29;
          v56 = v30;
        }
        v26 = &dest.m128i_i8[8LL - (_QWORD)v27];
        if ( v26 > (char *)&v54 )
          abort();
        memcpy(&dest, (char *)&src + (_QWORD)v27, 8LL - (_QWORD)v27);
      }
      result = sub_1E30AD0(dest.m128i_i8, v25, v26, v54.m128i_i8);
      break;
    case 2:
      v31 = *(_QWORD *)(a1 + 24);
      LOBYTE(v45) = 2;
      dest.m128i_i64[0] = v31;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35660((char *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 3:
      v6 = *(_QWORD *)(a1 + 24);
      LOBYTE(v45) = 3;
      dest.m128i_i64[0] = v6;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35730((char *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 4:
      v7 = *(_QWORD *)(a1 + 24);
      LOBYTE(v45) = 4;
      dest.m128i_i64[0] = v7;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35800((char *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 5:
      v5 = *(_DWORD *)(a1 + 24);
      LOBYTE(v45) = 5;
      dest.m128i_i32[0] = v5;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E358D0((char *)&v45, (int *)&src, dest.m128i_i32);
      break;
    case 6:
    case 7:
      v3 = *(int *)(a1 + 32);
      v4 = *(unsigned int *)(a1 + 8);
      LOBYTE(v44) = *(_BYTE *)a1;
      dest.m128i_i64[0] = v4 | (v3 << 32);
      LODWORD(src) = *(_DWORD *)(a1 + 24);
      LODWORD(v45) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E359A0((char *)&v44, (int *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 8:
      v8 = *(_DWORD *)(a1 + 24);
      LOBYTE(v45) = 8;
      dest.m128i_i32[0] = v8;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E358D0((char *)&v45, (int *)&src, dest.m128i_i32);
      break;
    case 9:
      v9 = *(_QWORD *)(a1 + 24);
      v10 = *(unsigned int *)(a1 + 8);
      LOBYTE(v44) = 9;
      dest.m128i_i64[0] = v9;
      src = v10 | ((__int64)*(int *)(a1 + 32) << 32);
      LODWORD(v45) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35A70((char *)&v44, (int *)&v45, &src, dest.m128i_i64);
      break;
    case 0xA:
      v11 = *(int *)(a1 + 32);
      v12 = *(unsigned int *)(a1 + 8);
      LOBYTE(v44) = 10;
      dest.m128i_i64[0] = v12 | (v11 << 32);
      src = *(_QWORD *)(a1 + 24);
      LODWORD(v45) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35B40((char *)&v44, (int *)&v45, &src, dest.m128i_i64);
      break;
    case 0xB:
      v13 = *(int *)(a1 + 32);
      v14 = *(unsigned int *)(a1 + 8);
      LOBYTE(v44) = 11;
      dest.m128i_i64[0] = v14 | (v13 << 32);
      src = *(_QWORD *)(a1 + 24);
      LODWORD(v45) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35C10((char *)&v44, (int *)&v45, &src, dest.m128i_i64);
      break;
    case 0xC:
    case 0xD:
      v1 = *(_QWORD *)(a1 + 24);
      LOBYTE(v45) = *(_BYTE *)a1;
      dest.m128i_i64[0] = v1;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35CE0((char *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 0xE:
      v15 = *(_QWORD *)(a1 + 24);
      LOBYTE(v45) = 14;
      dest.m128i_i64[0] = v15;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35DB0((char *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 0xF:
      v16 = *(_QWORD *)(a1 + 24);
      LOBYTE(v45) = 15;
      dest.m128i_i64[0] = v16;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35E80((char *)&v45, (int *)&src, dest.m128i_i64);
      break;
    case 0x10:
      v17 = *(_DWORD *)(a1 + 24);
      LOBYTE(v45) = 16;
      dest.m128i_i32[0] = v17;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35F50((char *)&v45, (int *)&src, dest.m128i_i32);
      break;
    case 0x11:
      v18 = *(_DWORD *)(a1 + 24);
      LOBYTE(v45) = 17;
      dest.m128i_i32[0] = v18;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E36020((char *)&v45, (int *)&src, dest.m128i_i32);
      break;
    case 0x12:
      v19 = *(_DWORD *)(a1 + 24);
      LOBYTE(v45) = 18;
      dest.m128i_i32[0] = v19;
      LODWORD(src) = (*(_DWORD *)a1 >> 8) & 0xFFF;
      result = sub_1E35F50((char *)&v45, (int *)&src, dest.m128i_i32);
      break;
  }
  return result;
}
