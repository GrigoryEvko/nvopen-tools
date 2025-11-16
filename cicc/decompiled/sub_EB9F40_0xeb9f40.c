// Function: sub_EB9F40
// Address: 0xeb9f40
//
__int64 __fastcall sub_EB9F40(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // rax
  const __m128i *v6; // rsi
  const char *v7; // r13
  int v8; // edx
  __m128i *v9; // rax
  _DWORD *v10; // rax
  __int64 v11; // r9
  int v12; // eax
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdi
  const char **v17; // rsi
  __int64 v18; // r12
  char v19; // r15
  __int64 v20; // rax
  const char *v21; // rdx
  __int64 v22; // rsi
  const char *v23; // rdi
  int v24; // eax
  int v25; // [rsp+0h] [rbp-150h]
  __int64 v26; // [rsp+8h] [rbp-148h]
  unsigned __int8 v27; // [rsp+18h] [rbp-138h]
  const char **v28; // [rsp+28h] [rbp-128h] BYREF
  __int64 v29; // [rsp+30h] [rbp-120h] BYREF
  __int64 v30; // [rsp+38h] [rbp-118h] BYREF
  __int64 v31; // [rsp+40h] [rbp-110h] BYREF
  __int64 v32; // [rsp+48h] [rbp-108h] BYREF
  const char *v33; // [rsp+50h] [rbp-100h] BYREF
  __int64 v34; // [rsp+58h] [rbp-F8h]
  const char *v35; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v36; // [rsp+68h] [rbp-E8h]
  __int16 v37; // [rsp+80h] [rbp-D0h]
  const char *v38; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+98h] [rbp-B8h]
  _BYTE v40[16]; // [rsp+A0h] [rbp-B0h] BYREF
  char v41; // [rsp+B0h] [rbp-A0h]
  char v42; // [rsp+B1h] [rbp-9Fh]

  v42 = 1;
  v32 = 0;
  v38 = "expected integer";
  v41 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v28, &v38) )
    return 1;
  v42 = 1;
  v38 = "expected integer";
  v41 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v29, &v38) )
    return 1;
  v42 = 1;
  v38 = "expected integer";
  v41 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v30, &v38) )
    return 1;
  v42 = 1;
  v38 = "expected integer";
  v41 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v31, &v38) )
    return 1;
  if ( (v31 & 4) == 0
    || (v42 = 1, v38 = "expected integer", v41 = 3, result = sub_ECE130(a1, &v32, &v38), !(_BYTE)result) )
  {
    v38 = v40;
    v39 = 0x800000000LL;
    if ( **(_DWORD **)(a1 + 48) != 46 )
    {
LABEL_25:
      v33 = 0;
      v34 = 0;
      if ( (unsigned __int8)sub_EB61F0(a1, (__int64 *)&v33) )
      {
        v35 = "expected identifier";
        v37 = 259;
        v17 = (const char **)sub_ECD690(a1 + 40);
        result = sub_ECDA70(a1, v17, &v35, 0, 0);
      }
      else
      {
        v16 = *(_QWORD *)(a1 + 224);
        v17 = &v35;
        v37 = 261;
        v35 = v33;
        v36 = v34;
        v18 = sub_E65280(v16, &v35);
        result = sub_ECE000(a1);
        if ( !(_BYTE)result )
        {
          v17 = v28;
          (*(void (__fastcall **)(_QWORD, const char **, __int64, __int64, __int64, __int64, const char **, __int64))(**(_QWORD **)(a1 + 232) + 1232LL))(
            *(_QWORD *)(a1 + 232),
            v28,
            v29,
            v30,
            v31,
            v32,
            &v38,
            v18);
          result = 0;
        }
      }
      if ( v38 != v40 )
      {
        v27 = result;
        _libc_free(v38, v17);
        return v27;
      }
      return result;
    }
    while ( 1 )
    {
      sub_EABFE0(a1);
      v12 = **(_DWORD **)(a1 + 48);
      if ( v12 == 4 )
        break;
      v3 = 0;
      if ( v12 == 10 )
        goto LABEL_24;
      LODWORD(v4) = 0;
LABEL_12:
      v5 = (unsigned int)v39;
      v6 = (const __m128i *)&v35;
      v36 = v3;
      LODWORD(v35) = v4;
      v7 = v38;
      v8 = v39;
      if ( (unsigned __int64)(unsigned int)v39 + 1 > HIDWORD(v39) )
      {
        if ( v38 > (const char *)&v35 || &v35 >= (const char **)&v38[16 * (unsigned int)v39] )
        {
          v26 = -1;
          v19 = 0;
        }
        else
        {
          v19 = 1;
          v26 = ((char *)&v35 - v38) >> 4;
        }
        v20 = sub_C8D7D0((__int64)&v38, (__int64)v40, (unsigned int)v39 + 1LL, 0x10u, (unsigned __int64 *)&v33, v11);
        v21 = v38;
        v7 = (const char *)v20;
        v22 = 16LL * (unsigned int)v39;
        v23 = &v38[v22];
        if ( v38 != &v38[v22] )
        {
          v22 += v20;
          do
          {
            if ( v20 )
            {
              *(_DWORD *)v20 = *(_DWORD *)v21;
              *(_QWORD *)(v20 + 8) = *((_QWORD *)v21 + 1);
            }
            v20 += 16;
            v21 += 16;
          }
          while ( v22 != v20 );
          v23 = v38;
        }
        v24 = (int)v33;
        if ( v23 != v40 )
        {
          v25 = (int)v33;
          _libc_free(v23, v22);
          v24 = v25;
        }
        HIDWORD(v39) = v24;
        v5 = (unsigned int)v39;
        v38 = v7;
        v6 = (const __m128i *)&v7[16 * v26];
        v8 = v39;
        if ( !v19 )
          v6 = (const __m128i *)&v35;
      }
      v9 = (__m128i *)&v7[16 * v5];
      if ( v9 )
      {
        *v9 = _mm_loadu_si128(v6);
        v8 = v39;
      }
      v10 = *(_DWORD **)(a1 + 48);
      LODWORD(v39) = v8 + 1;
      if ( *v10 != 46 )
        goto LABEL_25;
    }
    v13 = sub_ECD7B0(a1);
    if ( *(_DWORD *)(v13 + 32) <= 0x40u )
      v3 = *(_QWORD *)(v13 + 24);
    else
      v3 = **(_QWORD **)(v13 + 24);
    sub_EABFE0(a1);
    v14 = **(_DWORD **)(a1 + 48);
    if ( v14 == 10 )
    {
LABEL_24:
      sub_EABFE0(a1);
      v14 = **(_DWORD **)(a1 + 48);
    }
    LODWORD(v4) = 0;
    if ( v14 == 4 )
    {
      v15 = sub_ECD7B0(a1);
      if ( *(_DWORD *)(v15 + 32) <= 0x40u )
        v4 = *(_QWORD *)(v15 + 24);
      else
        v4 = **(_QWORD **)(v15 + 24);
      sub_EABFE0(a1);
    }
    goto LABEL_12;
  }
  return result;
}
