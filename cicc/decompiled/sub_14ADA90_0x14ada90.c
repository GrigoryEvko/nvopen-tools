// Function: sub_14ADA90
// Address: 0x14ada90
//
__int64 __fastcall sub_14ADA90(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  int v4; // eax
  __int64 *v5; // r12
  __int64 *v6; // r13
  char v7; // dl
  int v8; // eax
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 *v12; // rax
  __int64 *v13; // rsi
  __int64 *v14; // rcx
  __int64 *v15; // rax
  __int64 v16; // r8
  int v17; // eax
  __int64 v18; // rax
  _QWORD *v19; // rdi
  int v20; // eax
  __int64 *v21; // rdx
  __int64 v22; // rcx
  unsigned __int8 v23; // al
  __int64 *v24; // r8
  __int64 *v25; // rdi
  __int64 v26; // [rsp+0h] [rbp-170h]
  __int64 v27; // [rsp+18h] [rbp-158h]
  unsigned __int8 v29; // [rsp+28h] [rbp-148h]
  unsigned __int8 v30; // [rsp+28h] [rbp-148h]
  _QWORD *v31; // [rsp+30h] [rbp-140h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-138h]
  unsigned int v33; // [rsp+3Ch] [rbp-134h]
  _QWORD v34[4]; // [rsp+40h] [rbp-130h] BYREF
  __int64 *v35; // [rsp+60h] [rbp-110h] BYREF
  __int64 v36; // [rsp+68h] [rbp-108h]
  _BYTE v37[32]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v38; // [rsp+90h] [rbp-E0h] BYREF
  __int64 *v39; // [rsp+98h] [rbp-D8h]
  __int64 *v40; // [rsp+A0h] [rbp-D0h]
  __int64 v41; // [rsp+A8h] [rbp-C8h]
  int v42; // [rsp+B0h] [rbp-C0h]
  _BYTE v43[184]; // [rsp+B8h] [rbp-B8h] BYREF

  v27 = a2 + 16;
  v38 = 0;
  v41 = 16;
  v42 = 0;
  v33 = 4;
  v34[0] = a1;
  v39 = (__int64 *)v43;
  v40 = (__int64 *)v43;
  v31 = v34;
  v4 = 1;
  while ( 2 )
  {
    v32 = v4 - 1;
    v36 = 0x400000000LL;
    v35 = (__int64 *)v37;
    sub_14AD470(a1, (__int64)&v35, a3, 0, 6u);
    v5 = &v35[(unsigned int)v36];
    if ( v35 == v5 )
      goto LABEL_33;
    v6 = v35;
    do
    {
      while ( 1 )
      {
        v11 = *v6;
        v12 = v39;
        if ( v40 != v39 )
        {
LABEL_4:
          sub_16CCBA0(&v38, *v6);
          if ( !v7 )
            goto LABEL_11;
          goto LABEL_5;
        }
        v13 = &v39[HIDWORD(v41)];
        if ( v39 == v13 )
          break;
        v14 = 0;
        while ( v11 != *v12 )
        {
          if ( *v12 == -2 )
            v14 = v12;
          if ( v13 == ++v12 )
          {
            if ( !v14 )
              goto LABEL_48;
            *v14 = v11;
            --v42;
            v8 = *(unsigned __int8 *)(v11 + 16);
            ++v38;
            if ( (unsigned __int8)v8 > 0x17u )
              goto LABEL_6;
            goto LABEL_21;
          }
        }
LABEL_11:
        if ( v5 == ++v6 )
          goto LABEL_32;
      }
LABEL_48:
      if ( HIDWORD(v41) >= (unsigned int)v41 )
        goto LABEL_4;
      ++HIDWORD(v41);
      *v13 = v11;
      ++v38;
LABEL_5:
      v8 = *(unsigned __int8 *)(v11 + 16);
      if ( (unsigned __int8)v8 <= 0x17u )
      {
LABEL_21:
        if ( (_BYTE)v8 != 5 || *(_WORD *)(v11 + 18) != 46 )
          goto LABEL_7;
      }
      else
      {
LABEL_6:
        if ( v8 != 70 )
          goto LABEL_7;
      }
      if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
        v15 = *(__int64 **)(v11 - 8);
      else
        v15 = (__int64 *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      v16 = *v15;
      while ( 1 )
      {
        v17 = *(unsigned __int8 *)(v16 + 16);
        if ( (unsigned __int8)v17 <= 0x17u )
          break;
        v20 = v17 - 24;
        if ( v20 == 45 )
          goto LABEL_51;
LABEL_38:
        if ( v20 != 11 )
          goto LABEL_28;
        if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
          v21 = *(__int64 **)(v16 - 8);
        else
          v21 = (__int64 *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
        v22 = v21[3];
        v23 = *(_BYTE *)(v22 + 16);
        if ( v23 == 13 )
          goto LABEL_45;
        if ( v23 <= 0x17u )
        {
          if ( v23 != 5 || *(_WORD *)(v22 + 18) != 15 )
            goto LABEL_28;
          v16 = *v21;
        }
        else
        {
          if ( v23 != 39 && v23 != 77 )
            goto LABEL_28;
LABEL_45:
          v16 = *v21;
        }
      }
      if ( (_BYTE)v17 != 5 )
        goto LABEL_28;
      v20 = *(unsigned __int16 *)(v16 + 18);
      if ( v20 != 45 )
        goto LABEL_38;
LABEL_51:
      if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
        v24 = *(__int64 **)(v16 - 8);
      else
        v24 = (__int64 *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
      v16 = *v24;
LABEL_28:
      if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 15 )
      {
LABEL_7:
        result = sub_134E860(v11);
        if ( !(_BYTE)result )
        {
          v25 = v35;
          *(_DWORD *)(a2 + 8) = 0;
          if ( v25 == (__int64 *)v37 )
          {
            v19 = v31;
          }
          else
          {
            _libc_free((unsigned __int64)v25);
            v19 = v31;
            result = 0;
          }
          goto LABEL_59;
        }
        v10 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, v27, 0, 8);
          v10 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v10) = v11;
        ++*(_DWORD *)(a2 + 8);
        goto LABEL_11;
      }
      v18 = v32;
      if ( v32 >= v33 )
      {
        v26 = v16;
        sub_16CD150(&v31, v34, 0, 8);
        v18 = v32;
        v16 = v26;
      }
      ++v6;
      v31[v18] = v16;
      ++v32;
    }
    while ( v5 != v6 );
LABEL_32:
    v5 = v35;
LABEL_33:
    if ( v5 != (__int64 *)v37 )
      _libc_free((unsigned __int64)v5);
    v4 = v32;
    v19 = v31;
    if ( v32 )
    {
      a1 = v31[v32 - 1];
      continue;
    }
    break;
  }
  result = 1;
LABEL_59:
  if ( v19 != v34 )
  {
    v29 = result;
    _libc_free((unsigned __int64)v19);
    result = v29;
  }
  if ( v40 != v39 )
  {
    v30 = result;
    _libc_free((unsigned __int64)v40);
    return v30;
  }
  return result;
}
