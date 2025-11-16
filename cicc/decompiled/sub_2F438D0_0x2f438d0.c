// Function: sub_2F438D0
// Address: 0x2f438d0
//
int *__fastcall sub_2F438D0(__int64 a1, __int64 a2, int a3, unsigned int a4)
{
  int *result; // rax
  __int64 v6; // rdx
  unsigned int v9; // ecx
  int v10; // esi
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r15
  int v16; // r12d
  _BYTE *v17; // rsi
  _BYTE *v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rsi
  __int64 (__fastcall **v21)(_QWORD *, _DWORD *, int); // rdi
  __int64 v22; // rdx
  int v23; // r8d
  unsigned int v24; // [rsp+Ch] [rbp-124h]
  int *v26; // [rsp+18h] [rbp-118h]
  __int64 *v27; // [rsp+28h] [rbp-108h]
  unsigned __int16 v28; // [rsp+30h] [rbp-100h]
  unsigned __int16 v29; // [rsp+32h] [rbp-FEh]
  __int64 *v31; // [rsp+38h] [rbp-F8h]
  __m128i v32; // [rsp+40h] [rbp-F0h] BYREF
  __int64 (__fastcall *v33)(__m128i *, __m128i *, int); // [rsp+50h] [rbp-E0h] BYREF
  bool (__fastcall *v34)(_DWORD *, __int64); // [rsp+58h] [rbp-D8h]
  void (__fastcall *v35)(__int64 (__fastcall **)(__m128i *, __m128i *, int), __int64 (__fastcall **)(__m128i *, __m128i *, int), __int64); // [rsp+60h] [rbp-D0h]
  unsigned __int8 (__fastcall *v36)(__int64 (__fastcall **)(_QWORD *, _DWORD *, int), __int64); // [rsp+68h] [rbp-C8h]
  _QWORD v37[2]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v38[2]; // [rsp+80h] [rbp-B0h] BYREF
  void (__fastcall *v39)(_QWORD *, _QWORD *, __int64); // [rsp+90h] [rbp-A0h]
  __int64 v40; // [rsp+98h] [rbp-98h]
  __m128i v41; // [rsp+A0h] [rbp-90h] BYREF
  __int64 (__fastcall *v42[2])(__m128i *, __m128i *, int); // [rsp+B0h] [rbp-80h] BYREF
  void (__fastcall *v43)(__int64 (__fastcall **)(__m128i *, __m128i *, int), __int64 (__fastcall **)(__m128i *, __m128i *, int), __int64); // [rsp+C0h] [rbp-70h]
  unsigned __int8 (__fastcall *v44)(__int64 (__fastcall **)(_QWORD *, _DWORD *, int), __int64); // [rsp+C8h] [rbp-68h]
  __int64 v45; // [rsp+D0h] [rbp-60h]
  __int64 v46; // [rsp+D8h] [rbp-58h]
  _BYTE v47[16]; // [rsp+E0h] [rbp-50h] BYREF
  void (__fastcall *v48)(_QWORD *, _BYTE *, __int64); // [rsp+F0h] [rbp-40h]
  __int64 v49; // [rsp+F8h] [rbp-38h]

  result = (int *)*(unsigned int *)(a1 + 728);
  v6 = *(_QWORD *)(a1 + 712);
  if ( (_DWORD)result )
  {
    v9 = ((_DWORD)result - 1) & (37 * a3);
    v26 = (int *)(v6 + 32LL * v9);
    v10 = *v26;
    if ( *v26 == a3 )
    {
LABEL_3:
      result = (int *)(v6 + 32LL * (_QWORD)result);
      if ( v26 != result )
      {
        v28 = a4;
        v11 = (__int64 *)*((_QWORD *)v26 + 1);
        v31 = v11;
        v27 = &v11[v26[4]];
        if ( v27 != v11 )
        {
          v24 = a4;
          do
          {
            v12 = *v31;
            v13 = *(_QWORD *)(*v31 + 32);
            v14 = v13 + 40;
            if ( *(_WORD *)(*v31 + 68) != 14 )
            {
              v14 = v13 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF);
              v13 += 80;
            }
            if ( v14 != sub_2F41510(v13, v14, a3) )
            {
              v29 = v28;
              v15 = *(_QWORD *)(a2 + 8);
              if ( v12 != v15 )
              {
                v16 = 20;
                while ( (unsigned int)sub_2E8E710(v15, v24, *(_QWORD *)(a1 + 16), 0, 1) == -1 )
                {
                  if ( !--v16 )
                    break;
                  if ( !v15 )
                    BUG();
                  if ( (*(_BYTE *)v15 & 4) != 0 )
                  {
                    v15 = *(_QWORD *)(v15 + 8);
                    if ( v12 == v15 )
                      goto LABEL_18;
                  }
                  else
                  {
                    while ( (*(_BYTE *)(v15 + 44) & 8) != 0 )
                      v15 = *(_QWORD *)(v15 + 8);
                    v15 = *(_QWORD *)(v15 + 8);
                    if ( v12 == v15 )
                      goto LABEL_18;
                  }
                }
                v29 = 0;
              }
LABEL_18:
              v32.m128i_i32[0] = a3;
              v38[0] = 0;
              v34 = sub_2E85490;
              v33 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_2E854D0;
              sub_2E854D0(v37, &v32, 2);
              v38[1] = v34;
              v38[0] = v33;
              v17 = *(_BYTE **)(v12 + 32);
              if ( *(_WORD *)(v12 + 68) == 14 )
              {
                v18 = v17 + 40;
              }
              else
              {
                v18 = &v17[40 * (*(_DWORD *)(v12 + 40) & 0xFFFFFF)];
                v17 += 80;
              }
              sub_2F434D0(&v41, v17, v18, (__int64)v37);
              if ( v38[0] )
                ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v38[0])(v37, v37, 3);
              if ( v33 )
                v33(&v32, &v32, 3);
              v35 = 0;
              v32 = v41;
              if ( v43 )
              {
                v43(&v33, v42, 2);
                v36 = v44;
                v35 = v43;
              }
              v39 = 0;
              v37[0] = v45;
              v37[1] = v46;
              if ( v48 )
              {
                v48(v38, v47, 2);
                v40 = v49;
                v39 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v48;
              }
              while ( 1 )
              {
                v19 = v32.m128i_i64[0];
                if ( v32.m128i_i64[0] == v37[0] )
                  break;
                while ( 1 )
                {
                  v20 = v29;
                  v21 = (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v19;
                  sub_2EAB0C0(v19, v29);
                  if ( v29 )
                  {
                    v20 = 1;
                    v21 = (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v19;
                    sub_2EAB350(v19, 1);
                  }
                  v19 = v32.m128i_i64[0] + 40;
                  v32.m128i_i64[0] = v19;
                  if ( v19 != v32.m128i_i64[1] )
                    break;
LABEL_35:
                  if ( v37[0] == v19 )
                    goto LABEL_36;
                }
                while ( 1 )
                {
                  if ( !v35 )
                    sub_4263D6(v21, v20, v22);
                  v20 = v19;
                  v21 = (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))&v33;
                  if ( v36((__int64 (__fastcall **)(_QWORD *, _DWORD *, int))&v33, v19) )
                    break;
                  v19 = v32.m128i_i64[0] + 40;
                  v32.m128i_i64[0] = v19;
                  if ( v32.m128i_i64[1] == v19 )
                    goto LABEL_35;
                }
              }
LABEL_36:
              if ( v39 )
                v39(v38, v38, 3);
              if ( v35 )
                v35(&v33, &v33, 3);
              if ( v48 )
                v48(v47, v47, 3);
              if ( v43 )
                v43(v42, v42, 3);
            }
            ++v31;
          }
          while ( v27 != v31 );
        }
        result = v26;
        v26[4] = 0;
      }
    }
    else
    {
      v23 = 1;
      while ( v10 != -1 )
      {
        v9 = ((_DWORD)result - 1) & (v23 + v9);
        v26 = (int *)(v6 + 32LL * v9);
        v10 = *v26;
        if ( a3 == *v26 )
          goto LABEL_3;
        ++v23;
      }
    }
  }
  return result;
}
