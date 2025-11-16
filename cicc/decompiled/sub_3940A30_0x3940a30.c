// Function: sub_3940A30
// Address: 0x3940a30
//
__int64 __fastcall sub_3940A30(_QWORD *a1)
{
  __int64 result; // rax
  unsigned __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // rsi
  const __m128i *v15; // rdi
  __m128i *v16; // rcx
  const __m128i *i; // rax
  unsigned __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r13
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rdi
  __int64 v28; // [rsp+8h] [rbp-138h]
  __int64 v29; // [rsp+10h] [rbp-130h]
  __int64 v30; // [rsp+18h] [rbp-128h]
  __int64 v31; // [rsp+20h] [rbp-120h]
  __int64 v32; // [rsp+28h] [rbp-118h]
  __int64 v33; // [rsp+28h] [rbp-118h]
  __int64 v34; // [rsp+30h] [rbp-110h] BYREF
  char v35; // [rsp+40h] [rbp-100h]
  __int64 v36; // [rsp+50h] [rbp-F0h] BYREF
  char v37; // [rsp+60h] [rbp-E0h]
  __int64 v38; // [rsp+70h] [rbp-D0h] BYREF
  char v39; // [rsp+80h] [rbp-C0h]
  __int64 v40; // [rsp+90h] [rbp-B0h] BYREF
  char v41; // [rsp+A0h] [rbp-A0h]
  __int64 v42; // [rsp+B0h] [rbp-90h] BYREF
  char v43; // [rsp+C0h] [rbp-80h]
  unsigned __int64 v44; // [rsp+D0h] [rbp-70h] BYREF
  char v45; // [rsp+E0h] [rbp-60h]
  const __m128i *v46; // [rsp+F0h] [rbp-50h] BYREF
  const __m128i *v47; // [rsp+F8h] [rbp-48h]
  __int64 v48; // [rsp+100h] [rbp-40h]

  sub_393FF90((__int64)&v34, a1);
  if ( (v35 & 1) == 0 || (result = (unsigned int)v34, !(_DWORD)v34) )
  {
    sub_393FF90((__int64)&v36, a1);
    if ( (v37 & 1) == 0 || (result = (unsigned int)v36, !(_DWORD)v36) )
    {
      sub_393FF90((__int64)&v38, a1);
      if ( (v39 & 1) == 0 || (result = (unsigned int)v38, !(_DWORD)v38) )
      {
        sub_393FF90((__int64)&v40, a1);
        if ( (v41 & 1) == 0 || (result = (unsigned int)v40, !(_DWORD)v40) )
        {
          sub_393FF90((__int64)&v42, a1);
          if ( (v43 & 1) == 0 || (result = (unsigned int)v42, !(_DWORD)v42) )
          {
            sub_393FF90((__int64)&v44, a1);
            if ( (v45 & 1) == 0 || (result = (unsigned int)v44, !(_DWORD)v44) )
            {
              v46 = 0;
              v47 = 0;
              v48 = 0;
              if ( v44 )
              {
                LODWORD(v3) = 0;
                do
                {
                  v4 = sub_3940900(a1, (unsigned __int64 *)&v46);
                  v6 = v5;
                  v10 = sub_393D180((__int64)a1, (__int64)&v46, v5, v7, v8, v9);
                  if ( (_DWORD)v4 || v10 != v6 )
                  {
                    result = v4;
                    goto LABEL_14;
                  }
                  v3 = (unsigned int)(v3 + 1);
                }
                while ( v3 < v44 );
                v12 = (char *)v47 - (char *)v46;
                if ( v47 == v46 )
                {
                  v13 = 0;
                }
                else
                {
                  if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
                    sub_4261EA(a1, &v46, v11);
                  v13 = sub_22077B0((char *)v47 - (char *)v46);
                }
              }
              else
              {
                v12 = 0;
                v13 = 0;
              }
              v14 = (unsigned __int64)v46;
              v15 = v47;
              v16 = (__m128i *)v13;
              for ( i = v46; i != v15; i = (const __m128i *)((char *)i + 24) )
              {
                if ( v16 )
                {
                  *v16 = _mm_loadu_si128(i);
                  v16[1].m128i_i64[0] = i[1].m128i_i64[0];
                }
                v16 = (__m128i *)((char *)v16 + 24);
              }
              v18 = (unsigned __int64)i - v14;
              v19 = v40;
              v28 = v34;
              v29 = v36;
              v30 = v38;
              v31 = v40;
              v33 = v42;
              v20 = sub_22077B0(0x48u);
              v25 = v20;
              v26 = v13 + 24 * ((0xAAAAAAAAAAAAAABLL * (v18 >> 3)) & 0x1FFFFFFFFFFFFFFFLL);
              if ( v20 )
              {
                v21 = v13 + v12;
                *(_DWORD *)v20 = 1;
                v24 = v29;
                v23 = v30;
                *(_QWORD *)(v20 + 8) = v13;
                v19 = v31;
                v22 = v33;
                *(_QWORD *)(v20 + 16) = v26;
                *(_QWORD *)(v20 + 24) = v13 + v12;
                *(_QWORD *)(v20 + 32) = v28;
                *(_QWORD *)(v20 + 40) = v29;
                *(_QWORD *)(v20 + 48) = 0;
                *(_QWORD *)(v20 + 56) = v30;
                *(_DWORD *)(v20 + 64) = v31;
                *(_DWORD *)(v20 + 68) = v33;
              }
              else if ( v13 )
              {
                v19 = v12;
                j_j___libc_free_0(v13);
              }
              v27 = a1[7];
              a1[7] = v25;
              if ( v27 )
                sub_393DAE0(v27);
              sub_393D180(v27, v19, v21, v22, v23, v24);
              result = 0;
LABEL_14:
              if ( v46 )
              {
                v32 = result;
                j_j___libc_free_0((unsigned __int64)v46);
                return v32;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
