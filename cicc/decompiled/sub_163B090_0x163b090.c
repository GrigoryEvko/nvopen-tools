// Function: sub_163B090
// Address: 0x163b090
//
__int64 __fastcall sub_163B090(__int64 a1)
{
  __int64 v2; // rdi
  int v3; // r12d
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 result; // rax
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  const __m128i *v16; // rdi
  signed __int64 v17; // rsi
  __int64 v19; // r13
  __int64 v20; // rax
  _QWORD *v21; // rbx
  _BYTE *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rcx
  __int64 v30; // rsi
  _QWORD *v31; // rdx
  const __m128i *v32; // rsi
  const __m128i *v33; // r13
  const __m128i *v34; // r14
  const __m128i *v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // [rsp-A0h] [rbp-A0h]
  __int64 v40; // [rsp-90h] [rbp-90h] BYREF
  __int64 v41; // [rsp-88h] [rbp-88h] BYREF
  __int64 v42; // [rsp-80h] [rbp-80h] BYREF
  __int64 v43; // [rsp-78h] [rbp-78h] BYREF
  __int64 v44; // [rsp-70h] [rbp-70h] BYREF
  __int64 v45; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v46; // [rsp-60h] [rbp-60h] BYREF
  _QWORD *v47; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v48; // [rsp-50h] [rbp-50h] BYREF
  const __m128i *v49; // [rsp-48h] [rbp-48h] BYREF
  const __m128i *v50; // [rsp-40h] [rbp-40h]
  const __m128i *v51; // [rsp-38h] [rbp-38h]

  if ( !a1 )
    return 0;
  if ( *(_BYTE *)a1 != 4 )
    return 0;
  if ( *(_DWORD *)(a1 + 8) != 8 )
    return 0;
  v2 = *(_QWORD *)(a1 - 64);
  if ( !v2 || *(_BYTE *)v2 != 4 )
    return 0;
  if ( *(_DWORD *)(v2 + 8) == 2 )
  {
    if ( sub_163ADA0(v2, "SampleProfile") )
    {
      v3 = 1;
      goto LABEL_9;
    }
    v2 = *(_QWORD *)(a1 - 64);
    if ( !v2 || *(_BYTE *)v2 != 4 )
      return 0;
  }
  if ( *(_DWORD *)(v2 + 8) != 2 || !sub_163ADA0(v2, "InstrProf") )
    return 0;
  v3 = 0;
LABEL_9:
  v4 = *(_QWORD *)(a1 + 8 * (1LL - *(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)v4 != 4 )
    return 0;
  if ( *(_DWORD *)(v4 + 8) != 2 )
    return 0;
  if ( !(unsigned __int8)sub_163ACF0(v4, "TotalCount", &v41) )
    return 0;
  v5 = *(_QWORD *)(a1 + 8 * (2LL - *(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)v5 != 4 )
    return 0;
  if ( *(_DWORD *)(v5 + 8) != 2 )
    return 0;
  if ( !(unsigned __int8)sub_163ACF0(v5, "MaxCount", &v44) )
    return 0;
  v6 = *(_QWORD *)(a1 + 8 * (3LL - *(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)v6 != 4 )
    return 0;
  if ( *(_DWORD *)(v6 + 8) != 2 )
    return 0;
  if ( !(unsigned __int8)sub_163ACF0(v6, "MaxInternalCount", &v45) )
    return 0;
  v7 = *(_QWORD *)(a1 + 8 * (4LL - *(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)v7 != 4 )
    return 0;
  if ( *(_DWORD *)(v7 + 8) != 2 )
    return 0;
  if ( !(unsigned __int8)sub_163ACF0(v7, "MaxFunctionCount", &v43) )
    return 0;
  v8 = *(_QWORD *)(a1 + 8 * (5LL - *(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)v8 != 4 )
    return 0;
  if ( *(_DWORD *)(v8 + 8) != 2 )
    return 0;
  if ( !(unsigned __int8)sub_163ACF0(v8, "NumCounts", &v40) )
    return 0;
  v9 = *(_QWORD *)(a1 + 8 * (6LL - *(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)v9 != 4 || *(_DWORD *)(v9 + 8) != 2 || !(unsigned __int8)sub_163ACF0(v9, "NumFunctions", &v42) )
    return 0;
  v10 = *(unsigned int *)(a1 + 8);
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v11 = *(_QWORD *)(a1 + 8 * (7 - v10));
  result = 0;
  if ( *(_BYTE *)v11 == 4 && *(_DWORD *)(v11 + 8) == 2 )
  {
    v13 = *(_BYTE **)(v11 - 16);
    if ( !*v13 )
    {
      v14 = sub_161E970((__int64)v13);
      if ( v15 == 15 )
      {
        if ( *(_QWORD *)v14 != 0x64656C6961746544LL
          || *(_DWORD *)(v14 + 8) != 1835890003
          || *(_WORD *)(v14 + 12) != 29281
          || *(_BYTE *)(v14 + 14) != 121 )
        {
          v16 = v49;
          v17 = (char *)v51 - (char *)v49;
          goto LABEL_32;
        }
        v19 = *(_QWORD *)(v11 + 8 * (1LL - *(unsigned int *)(v11 + 8)));
        if ( *(_BYTE *)v19 != 4 )
        {
          v16 = v49;
          result = 0;
          v17 = (char *)v51 - (char *)v49;
LABEL_33:
          if ( v16 )
          {
            v39 = result;
            j_j___libc_free_0(v16, v17);
            return v39;
          }
          return result;
        }
        v20 = 8LL * *(unsigned int *)(v19 + 8);
        v21 = (_QWORD *)(v19 - v20);
        if ( v19 == v19 - v20 )
        {
LABEL_67:
          v33 = v49;
          v34 = v50;
          v49 = 0;
          v35 = v51;
          v50 = 0;
          v51 = 0;
          result = sub_22077B0(72);
          if ( result )
          {
            v36 = v41;
            *(_DWORD *)result = v3;
            *(_QWORD *)(result + 8) = v33;
            *(_QWORD *)(result + 32) = v36;
            v37 = v44;
            *(_QWORD *)(result + 16) = v34;
            *(_QWORD *)(result + 40) = v37;
            v38 = v45;
            *(_QWORD *)(result + 24) = v35;
            *(_QWORD *)(result + 48) = v38;
            *(_QWORD *)(result + 56) = v43;
            *(_DWORD *)(result + 64) = v40;
            *(_DWORD *)(result + 68) = v42;
          }
          else if ( v33 )
          {
            j_j___libc_free_0(v33, (char *)v35 - (char *)v33);
            result = 0;
          }
          v16 = v49;
          v17 = (char *)v51 - (char *)v49;
          goto LABEL_33;
        }
        while ( 1 )
        {
          v22 = (_BYTE *)*v21;
          if ( *(_BYTE *)*v21 != 4 || *((_DWORD *)v22 + 2) != 3 )
            break;
          v23 = *((_QWORD *)v22 - 3);
          v24 = *((_QWORD *)v22 - 2);
          if ( *(_BYTE *)v23 != 1 )
            v23 = 0;
          if ( *(_BYTE *)v24 != 1 )
            break;
          v25 = *((_QWORD *)v22 - 1);
          if ( *(_BYTE *)v25 != 1 || !v23 )
            break;
          v26 = *(_QWORD *)(v25 + 136);
          v27 = *(_QWORD **)(v26 + 24);
          if ( *(_DWORD *)(v26 + 32) > 0x40u )
            v27 = (_QWORD *)*v27;
          v28 = *(_QWORD *)(v24 + 136);
          v46 = v27;
          v29 = *(_QWORD **)(v28 + 24);
          if ( *(_DWORD *)(v28 + 32) > 0x40u )
            v29 = (_QWORD *)*v29;
          v30 = *(_QWORD *)(v23 + 136);
          v47 = v29;
          v31 = *(_QWORD **)(v30 + 24);
          if ( *(_DWORD *)(v30 + 32) > 0x40u )
            v31 = (_QWORD *)*v31;
          v48 = v31;
          v32 = v50;
          if ( v50 == v51 )
          {
            sub_163AED0(&v49, v50, &v48, &v47, &v46);
          }
          else
          {
            if ( v50 )
            {
              v50->m128i_i32[0] = (int)v31;
              v32->m128i_i64[1] = (__int64)v29;
              v32[1].m128i_i64[0] = (__int64)v27;
            }
            v50 = (const __m128i *)((char *)v50 + 24);
          }
          if ( (_QWORD *)v19 == ++v21 )
            goto LABEL_67;
        }
      }
      v16 = v49;
      v17 = (char *)v51 - (char *)v49;
LABEL_32:
      result = 0;
      goto LABEL_33;
    }
  }
  return result;
}
