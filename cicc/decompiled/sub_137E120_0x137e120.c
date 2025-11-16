// Function: sub_137E120
// Address: 0x137e120
//
__int64 __fastcall sub_137E120(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 *v10; // r8
  __int64 result; // rax
  __int64 *v12; // rdi
  __int64 *v13; // rcx
  char v14; // dl
  __int64 v15; // rdx
  __int64 v16; // r11
  _QWORD *v17; // rax
  _QWORD *v18; // r10
  __int64 v19; // r13
  __int64 *v20; // rsi
  __int64 *v21; // rax
  bool v22; // zf
  __int64 v23; // rdi
  int v24; // r12d
  __int64 v25; // r13
  __int64 v26; // rax
  unsigned int v27; // r14d
  __int64 *v28; // r15
  unsigned int v29; // edx
  __int64 v30; // rax
  int v31; // eax
  int v32; // r9d
  _BYTE v33[12]; // [rsp+14h] [rbp-17Ch]
  __int64 v34; // [rsp+20h] [rbp-170h]
  unsigned __int8 v36; // [rsp+28h] [rbp-168h]
  __int64 v37; // [rsp+30h] [rbp-160h] BYREF
  __int64 *v38; // [rsp+38h] [rbp-158h]
  __int64 *v39; // [rsp+40h] [rbp-150h]
  __int64 v40; // [rsp+48h] [rbp-148h]
  int v41; // [rsp+50h] [rbp-140h]
  _BYTE v42[312]; // [rsp+58h] [rbp-138h] BYREF

  v34 = a3;
  *(_QWORD *)&v33[4] = a4;
  if ( a3 )
  {
    v5 = *(unsigned int *)(a3 + 48);
    if ( (_DWORD)v5 )
    {
      v6 = *(_QWORD *)(v34 + 32);
      v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
      {
LABEL_4:
        if ( v8 != (__int64 *)(v6 + 16 * v5) && v8[1] )
          goto LABEL_7;
      }
      else
      {
        v31 = 1;
        while ( v9 != -8 )
        {
          v32 = v31 + 1;
          v7 = (v5 - 1) & (v31 + v7);
          v8 = (__int64 *)(v6 + 16LL * v7);
          v9 = *v8;
          if ( a2 == *v8 )
            goto LABEL_4;
          v31 = v32;
        }
      }
    }
  }
  v34 = 0;
LABEL_7:
  v10 = (__int64 *)v42;
  v37 = 0;
  LODWORD(result) = *(_DWORD *)(a1 + 8);
  v38 = (__int64 *)v42;
  v12 = (__int64 *)v42;
  v39 = (__int64 *)v42;
  v40 = 32;
  v41 = 0;
  *(_DWORD *)v33 = 32;
  while ( 1 )
  {
    v13 = (__int64 *)(unsigned int)result;
    v19 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a1 + 8) = result - 1;
    if ( v12 != v10 )
      goto LABEL_8;
    v20 = &v12[HIDWORD(v40)];
    if ( v20 != v12 )
    {
      v21 = v12;
      v13 = 0;
      while ( v19 != *v21 )
      {
        if ( *v21 == -2 )
          v13 = v21;
        if ( v20 == ++v21 )
        {
          if ( !v13 )
            goto LABEL_45;
          v15 = a2;
          *v13 = v19;
          --v41;
          v12 = v39;
          ++v37;
          v10 = v38;
          if ( a2 != v19 )
            goto LABEL_10;
          goto LABEL_43;
        }
      }
LABEL_26:
      result = *(unsigned int *)(a1 + 8);
      goto LABEL_18;
    }
LABEL_45:
    if ( HIDWORD(v40) < (unsigned int)v40 )
    {
      ++HIDWORD(v40);
      *v20 = v19;
      v10 = v38;
      ++v37;
      v12 = v39;
    }
    else
    {
LABEL_8:
      sub_16CCBA0(&v37, v19);
      v12 = v39;
      v10 = v38;
      if ( !v14 )
        goto LABEL_26;
    }
    v15 = a2;
    if ( a2 == v19 )
    {
LABEL_43:
      result = 1;
      goto LABEL_38;
    }
LABEL_10:
    if ( v34 )
    {
      result = sub_15CC8F0(v34, v19, v15, v13, v10);
      if ( (_BYTE)result )
        break;
    }
    if ( !*(_QWORD *)&v33[4] )
    {
      v22 = *(_DWORD *)v33 == 1;
      *(_QWORD *)v33 = (unsigned int)(*(_DWORD *)v33 - 1);
      if ( v22 )
        goto LABEL_37;
LABEL_28:
      v23 = sub_157EBA0(v19);
      if ( v23 )
      {
        v24 = sub_15F4D60(v23);
        v25 = sub_157EBA0(v19);
        v26 = *(unsigned int *)(a1 + 8);
        if ( v24 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v26 )
        {
          sub_16CD150(a1, a1 + 16, v24 + v26, 8);
          v26 = *(unsigned int *)(a1 + 8);
        }
        v27 = 0;
        v28 = (__int64 *)(*(_QWORD *)a1 + 8 * v26);
        v29 = v26 + v24;
        if ( v24 )
        {
          do
          {
            v30 = sub_15F4DF0(v25, v27);
            if ( v28 )
              *v28 = v30;
            ++v27;
            ++v28;
          }
          while ( v24 != v27 );
          v29 = v24 + *(_DWORD *)(a1 + 8);
        }
      }
      else
      {
        v29 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v29;
      v12 = v39;
      result = v29;
      v10 = v38;
      goto LABEL_18;
    }
    sub_137D930(*(__int64 *)&v33[4], v19);
    v17 = sub_137D930(v16, a2);
    if ( v18 && v18 == v17 || (--*(_DWORD *)v33, !*(_DWORD *)v33) )
    {
LABEL_37:
      v12 = v39;
      v10 = v38;
      result = 1;
      goto LABEL_38;
    }
    if ( !v18 )
      goto LABEL_28;
    sub_13F9EC0(v18, a1);
    result = *(unsigned int *)(a1 + 8);
    v12 = v39;
    v10 = v38;
LABEL_18:
    if ( !(_DWORD)result )
      goto LABEL_38;
  }
  v12 = v39;
  v10 = v38;
LABEL_38:
  if ( v12 != v10 )
  {
    v36 = result;
    _libc_free((unsigned __int64)v12);
    return v36;
  }
  return result;
}
