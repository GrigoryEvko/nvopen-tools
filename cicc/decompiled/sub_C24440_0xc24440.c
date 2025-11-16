// Function: sub_C24440
// Address: 0xc24440
//
__int64 __fastcall sub_C24440(const __m128i **a1)
{
  __int64 result; // rax
  const __m128i **v3; // rsi
  unsigned __int64 v4; // r13
  __int64 v5; // rbx
  __int64 (__fastcall ***v6)(); // rdx
  __int64 (__fastcall ***v7)(); // r12
  __int64 (__fastcall ***v8)(); // rax
  __int64 v9; // r14
  __int64 v10; // r13
  int v11; // r12d
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  const __m128i *v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __m128i *v19; // rsi
  const __m128i *v20; // r8
  const __m128i *v21; // r9
  __m128i *v22; // rdi
  const __m128i *i; // rax
  __int64 v24; // rdi
  __int64 v25; // [rsp+8h] [rbp-128h]
  __int64 v26; // [rsp+18h] [rbp-118h]
  __int64 v27; // [rsp+18h] [rbp-118h]
  __int64 v28; // [rsp+20h] [rbp-110h] BYREF
  char v29; // [rsp+30h] [rbp-100h]
  __int64 v30; // [rsp+40h] [rbp-F0h] BYREF
  char v31; // [rsp+50h] [rbp-E0h]
  _QWORD v32[4]; // [rsp+60h] [rbp-D0h] BYREF
  _DWORD v33[8]; // [rsp+80h] [rbp-B0h] BYREF
  _DWORD v34[8]; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD v35[4]; // [rsp+C0h] [rbp-70h] BYREF
  const __m128i *v36; // [rsp+E0h] [rbp-50h] BYREF
  const __m128i *v37; // [rsp+E8h] [rbp-48h]
  __int64 v38; // [rsp+F0h] [rbp-40h]

  sub_C21E40((__int64)&v28, a1);
  if ( (v29 & 1) == 0 || (result = (unsigned int)v28, !(_DWORD)v28) )
  {
    sub_C21E40((__int64)&v30, a1);
    if ( (v31 & 1) == 0 || (result = (unsigned int)v30, !(_DWORD)v30) )
    {
      sub_C21E40((__int64)v32, a1);
      result = sub_C21E20(v32);
      if ( !(_DWORD)result )
      {
        sub_C21E40((__int64)v33, a1);
        result = sub_C21E20(v33);
        if ( !(_DWORD)result )
        {
          sub_C21E40((__int64)v34, a1);
          result = sub_C21E20(v34);
          if ( !(_DWORD)result )
          {
            v3 = a1;
            sub_C21E40((__int64)v35, a1);
            result = sub_C21E20(v35);
            if ( !(_DWORD)result )
            {
              v36 = 0;
              v37 = 0;
              v38 = 0;
              if ( v35[0] )
              {
                LODWORD(v4) = 0;
                while ( 1 )
                {
                  v3 = &v36;
                  v5 = sub_C24310(a1, (__int64)&v36);
                  v7 = v6;
                  v8 = sub_C1AFD0();
                  if ( (_DWORD)v5 || v8 != v7 )
                    break;
                  v4 = (unsigned int)(v4 + 1);
                  if ( v4 >= v35[0] )
                    goto LABEL_18;
                }
                result = v5;
              }
              else
              {
LABEL_18:
                v9 = v30;
                v10 = v32[0];
                v11 = v33[0];
                v27 = v28;
                v12 = v34[0];
                v13 = sub_22077B0(88);
                v15 = v13;
                if ( v13 )
                {
                  *(_DWORD *)v13 = 2;
                  v16 = v37;
                  *(_QWORD *)(v13 + 8) = 0;
                  *(_QWORD *)(v13 + 16) = 0;
                  *(_QWORD *)(v13 + 24) = 0;
                  v17 = (char *)v16 - (char *)v36;
                  if ( v17 )
                  {
                    if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
                      sub_4261EA(v17, v3, v13, v14);
                    v25 = v13;
                    v18 = sub_22077B0(v17);
                    v15 = v25;
                    v19 = (__m128i *)v18;
                  }
                  else
                  {
                    v19 = 0;
                  }
                  v20 = v36;
                  v21 = v37;
                  *(_QWORD *)(v15 + 8) = v19;
                  *(_QWORD *)(v15 + 24) = (char *)v19 + v17;
                  v22 = v19;
                  *(_QWORD *)(v15 + 16) = v19;
                  for ( i = v20; i != v21; i = (const __m128i *)((char *)i + 24) )
                  {
                    if ( v22 )
                    {
                      *v22 = _mm_loadu_si128(i);
                      v22[1].m128i_i64[0] = i[1].m128i_i64[0];
                    }
                    v22 = (__m128i *)((char *)v22 + 24);
                  }
                  *(_QWORD *)(v15 + 40) = v9;
                  *(_QWORD *)(v15 + 48) = 0;
                  *(_QWORD *)(v15 + 56) = v10;
                  *(_DWORD *)(v15 + 64) = v11;
                  *(_DWORD *)(v15 + 68) = v12;
                  *(_BYTE *)(v15 + 72) = 0;
                  *(_QWORD *)(v15 + 80) = 0;
                  *(_QWORD *)(v15 + 16) = (char *)v19
                                        + 24
                                        * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)i - (char *)v20) >> 3))
                                         & 0x1FFFFFFFFFFFFFFFLL);
                  *(_QWORD *)(v15 + 32) = v27;
                }
                v24 = (__int64)a1[10];
                a1[10] = (const __m128i *)v15;
                if ( v24 )
                  sub_C1F020(v24);
                sub_C1AFD0();
                result = 0;
              }
              if ( v36 )
              {
                v26 = result;
                j_j___libc_free_0(v36, v38 - (_QWORD)v36);
                return v26;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
