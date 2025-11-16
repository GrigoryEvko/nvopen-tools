// Function: sub_31B9490
// Address: 0x31b9490
//
__int64 __fastcall sub_31B9490(__int64 a1, __int64 **a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r13
  __int64 v4; // rax
  __m128i *(__fastcall *v5)(__m128i *, _QWORD *); // rdx
  void (__fastcall *v6)(__m128i *, _QWORD *, __int64, _QWORD); // r12
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __m128i v10; // kr00_16
  __m128i *(__fastcall *v11)(__m128i *, __int64); // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r12
  bool v16; // al
  __int64 *v17; // r12
  __int64 **v18; // rbx
  __int64 result; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // ecx
  __int64 **v24; // rdx
  __int64 *v25; // r8
  __int64 v26; // rax
  __m128i *(__fastcall *v27)(__m128i *, __int64 *); // rdx
  void (__fastcall *v28)(__m128i *, __int64 *, __int64, _QWORD); // r15
  __int64 (__fastcall *v29)(__int64); // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __m128i v32; // kr20_16
  __m128i *(__fastcall *v33)(__m128i *, __int64); // rdx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // r12
  bool v38; // al
  __int64 v39; // rax
  __int64 v40; // rdi
  unsigned int v41; // esi
  __int64 *v42; // rcx
  __int64 v43; // r9
  __int64 v44; // rcx
  __int64 *v45; // rax
  bool v46; // al
  __int64 *v47; // rax
  bool v48; // al
  __int64 *v49; // rax
  __int64 *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  unsigned int v53; // esi
  __int64 *v54; // rcx
  __int64 v55; // r10
  __int64 v56; // rax
  int v57; // ecx
  int v58; // edx
  int v59; // edx
  int v60; // r9d
  int v61; // ecx
  int v62; // edx
  __int64 **v63; // rdx
  __int64 *v65; // [rsp+8h] [rbp-118h]
  __int64 v66; // [rsp+10h] [rbp-110h]
  __int64 v67; // [rsp+10h] [rbp-110h]
  __int64 v68; // [rsp+10h] [rbp-110h]
  __int64 **v69; // [rsp+18h] [rbp-108h]
  __int64 v70; // [rsp+28h] [rbp-F8h]
  __int64 *v71; // [rsp+28h] [rbp-F8h]
  __m128i v72; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v73; // [rsp+40h] [rbp-E0h]
  _QWORD v74[4]; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v75; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD v76[4]; // [rsp+90h] [rbp-90h] BYREF
  __m128i v77; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v78; // [rsp+C0h] [rbp-60h]
  __m128i v79; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v80; // [rsp+E0h] [rbp-40h]

  v2 = (__int64)a2[1];
  v3 = *a2;
  v69 = a2;
  v66 = v2;
  if ( v2 )
    v66 = sub_318B4B0(v2);
  for ( ; v3 != (_QWORD *)v66; v3 = (_QWORD *)sub_318B4B0((__int64)v3) )
  {
    v4 = *v3;
    v5 = *(__m128i *(__fastcall **)(__m128i *, _QWORD *))(*v3 + 40LL);
    if ( v5 == sub_3185D40 )
    {
      v6 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(v4 + 16);
      v7 = *(__int64 (__fastcall **)(__int64))(v4 + 64);
      if ( v7 == sub_3184E90 )
      {
        v8 = v3[2];
        v9 = 0;
        if ( (unsigned __int8)(*(_BYTE *)v8 - 22) > 6u )
          v9 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
      }
      else
      {
        v9 = (unsigned int)v7((__int64)v3);
      }
      v6(&v79, v3, v9, 0);
      v10 = v79;
    }
    else
    {
      v5(&v75, v3);
      v10 = v75;
    }
    v11 = *(__m128i *(__fastcall **)(__m128i *, __int64))(*v3 + 32LL);
    if ( v11 == sub_3184E50 )
    {
      (*(void (__fastcall **)(__m128i *, _QWORD *, _QWORD, _QWORD))(*v3 + 16LL))(&v79, v3, 0, 0);
      v13 = v79.m128i_i64[1];
      v12 = v79.m128i_i64[0];
      v14 = v80;
    }
    else
    {
      v11((__m128i *)v74, (__int64)v3);
      v12 = v74[0];
      v13 = v74[1];
      v14 = v74[2];
    }
    v77.m128i_i64[0] = v12;
    v77.m128i_i64[1] = v13;
    v78 = v14;
    while ( __PAIR128__(v77.m128i_u64[1], v12) != *(_OWORD *)&v10 )
    {
      sub_318E780(&v79, &v77);
      v15 = sub_318E5D0((__int64)&v79);
      v16 = sub_318B630(v15);
      if ( v15 )
      {
        if ( v16 )
        {
          v70 = sub_318B4F0(v15);
          if ( v70 == sub_318B4F0((__int64)v3) )
          {
            v49 = *v69;
            if ( *v69 )
            {
              if ( (__int64 *)v15 == v49 || sub_B445A0(v49[2], *(_QWORD *)(v15 + 16)) )
              {
                v50 = v69[1];
                if ( (__int64 *)v15 == v50 || sub_B445A0(*(_QWORD *)(v15 + 16), v50[2]) )
                {
                  v51 = *(_QWORD *)(a1 + 8);
                  v52 = *(unsigned int *)(a1 + 24);
                  if ( (_DWORD)v52 )
                  {
                    v53 = (v52 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
                    v54 = (__int64 *)(v51 + 16LL * v53);
                    v55 = *v54;
                    if ( v15 == *v54 )
                    {
LABEL_60:
                      if ( v54 != (__int64 *)(v51 + 16 * v52) )
                      {
                        v56 = v54[1];
                        if ( v56 )
                          ++*(_DWORD *)(v56 + 20);
                      }
                    }
                    else
                    {
                      v61 = 1;
                      while ( v55 != -4096 )
                      {
                        v62 = v61 + 1;
                        v53 = (v52 - 1) & (v61 + v53);
                        v54 = (__int64 *)(v51 + 16LL * v53);
                        v55 = *v54;
                        if ( v15 == *v54 )
                          goto LABEL_60;
                        v61 = v62;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      sub_318E7A0((__int64)&v77);
      v12 = v77.m128i_i64[0];
    }
  }
  v17 = *(__int64 **)(a1 + 32);
  v18 = (__int64 **)(a1 + 32);
  if ( v17 )
  {
    if ( sub_B445A0(v69[1][2], v17[2]) )
    {
      v17 = *(__int64 **)(a1 + 32);
    }
    else
    {
      v63 = v69;
      v69 = (__int64 **)(a1 + 32);
      v17 = *v63;
      v18 = v63;
    }
  }
  result = (__int64)v18[1];
  v71 = (__int64 *)result;
  if ( result )
  {
    result = sub_318B4B0(result);
    v71 = (__int64 *)result;
  }
  if ( v17 != v71 )
  {
    v20 = a1;
    while ( 1 )
    {
      v21 = *(unsigned int *)(v20 + 24);
      v22 = *(_QWORD *)(v20 + 8);
      if ( !(_DWORD)v21 )
        goto LABEL_83;
      v23 = (v21 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v24 = (__int64 **)(v22 + 16LL * v23);
      v25 = *v24;
      if ( *v24 != v17 )
        break;
LABEL_28:
      if ( v24 == (__int64 **)(v22 + 16 * v21) )
        goto LABEL_83;
      if ( !*((_BYTE *)v24[1] + 24) )
      {
        v26 = *v17;
        v27 = *(__m128i *(__fastcall **)(__m128i *, __int64 *))(*v17 + 40);
        if ( v27 == sub_3185D40 )
        {
          v28 = *(void (__fastcall **)(__m128i *, __int64 *, __int64, _QWORD))(v26 + 16);
          v29 = *(__int64 (__fastcall **)(__int64))(v26 + 64);
          if ( v29 == sub_3184E90 )
          {
            v30 = v17[2];
            v31 = 0;
            if ( (unsigned __int8)(*(_BYTE *)v30 - 22) > 6u )
              v31 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
          }
          else
          {
            v31 = (unsigned int)v29((__int64)v17);
          }
          v28(&v79, v17, v31, 0);
          v32 = v79;
        }
        else
        {
          v27(&v77, v17);
          v32 = v77;
        }
        v33 = *(__m128i *(__fastcall **)(__m128i *, __int64))(*v17 + 32);
        if ( v33 == sub_3184E50 )
        {
          (*(void (__fastcall **)(__m128i *, __int64 *, _QWORD, _QWORD))(*v17 + 16))(&v79, v17, 0, 0);
          v35 = v79.m128i_i64[1];
          v34 = v79.m128i_i64[0];
          v36 = v80;
        }
        else
        {
          v33((__m128i *)v76, (__int64)v17);
          v34 = v76[0];
          v35 = v76[1];
          v36 = v76[2];
        }
        v72.m128i_i64[0] = v34;
        v72.m128i_i64[1] = v35;
        v73 = v36;
        v65 = v17;
        while ( __PAIR128__(v72.m128i_u64[1], v34) != *(_OWORD *)&v32 )
        {
          sub_318E780(&v79, &v72);
          v37 = sub_318E5D0((__int64)&v79);
          v38 = sub_318B630(v37);
          if ( v37 )
          {
            if ( v38 )
            {
              v39 = *(unsigned int *)(v20 + 24);
              v40 = *(_QWORD *)(v20 + 8);
              if ( (_DWORD)v39 )
              {
                v41 = (v39 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
                v42 = (__int64 *)(v40 + 16LL * v41);
                v43 = *v42;
                if ( v37 == *v42 )
                {
LABEL_42:
                  if ( v42 != (__int64 *)(v40 + 16 * v39) )
                  {
                    v44 = v42[1];
                    if ( v44 )
                    {
                      v45 = *v69;
                      if ( *v69 )
                      {
                        if ( (__int64 *)v37 == v45
                          || (v67 = v44, v46 = sub_B445A0(v45[2], *(_QWORD *)(v37 + 16)), v44 = v67, v46) )
                        {
                          v47 = v69[1];
                          if ( (__int64 *)v37 == v47
                            || (v68 = v44, v48 = sub_B445A0(*(_QWORD *)(v37 + 16), v47[2]), v44 = v68, v48) )
                          {
                            ++*(_DWORD *)(v44 + 20);
                          }
                        }
                      }
                    }
                  }
                }
                else
                {
                  v57 = 1;
                  while ( v43 != -4096 )
                  {
                    v58 = v57 + 1;
                    v41 = (v39 - 1) & (v57 + v41);
                    v42 = (__int64 *)(v40 + 16LL * v41);
                    v43 = *v42;
                    if ( v37 == *v42 )
                      goto LABEL_42;
                    v57 = v58;
                  }
                }
              }
            }
          }
          sub_318E7A0((__int64)&v72);
          v34 = v72.m128i_i64[0];
        }
        v17 = v65;
      }
      result = sub_318B4B0((__int64)v17);
      v17 = (__int64 *)result;
      if ( (__int64 *)result == v71 )
        return result;
    }
    v59 = 1;
    while ( v25 != (__int64 *)-4096LL )
    {
      v60 = v59 + 1;
      v23 = (v21 - 1) & (v59 + v23);
      v24 = (__int64 **)(v22 + 16LL * v23);
      v25 = *v24;
      if ( *v24 == v17 )
        goto LABEL_28;
      v59 = v60;
    }
LABEL_83:
    BUG();
  }
  return result;
}
