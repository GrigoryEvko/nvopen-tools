// Function: sub_38A1B50
// Address: 0x38a1b50
//
__int64 __fastcall sub_38A1B50(__int64 **a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v8; // r14
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  char *v15; // rax
  __int64 v16; // rsi
  int v17; // r13d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rbx
  __int64 v23; // r14
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 v26; // rdx
  _QWORD *v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // r12
  int v34; // eax
  __int64 v35; // rax
  int v36; // edx
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // [rsp+8h] [rbp-1B8h]
  __int64 v40; // [rsp+10h] [rbp-1B0h]
  char v41; // [rsp+27h] [rbp-199h]
  __int64 v42; // [rsp+38h] [rbp-188h]
  __int64 *v43; // [rsp+38h] [rbp-188h]
  unsigned int v44; // [rsp+38h] [rbp-188h]
  __int64 v45; // [rsp+48h] [rbp-178h] BYREF
  __int64 v46; // [rsp+50h] [rbp-170h] BYREF
  __int64 v47; // [rsp+58h] [rbp-168h] BYREF
  __int64 v48[2]; // [rsp+60h] [rbp-160h] BYREF
  __int16 v49; // [rsp+70h] [rbp-150h]
  const char *v50; // [rsp+80h] [rbp-140h] BYREF
  __int64 v51; // [rsp+88h] [rbp-138h]
  _BYTE v52[304]; // [rsp+90h] [rbp-130h] BYREF

  v52[1] = 1;
  v8 = (unsigned __int64)a1[7];
  v45 = 0;
  v50 = "expected type";
  v52[0] = 3;
  if ( (unsigned __int8)sub_3891B00((__int64)a1, &v45, (__int64)&v50, 0) )
    return 1;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 6, "expected '[' in phi value list") )
    return 1;
  if ( (unsigned __int8)sub_38A1070(a1, v45, &v46, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 4, "expected ',' after insertelement value") )
    return 1;
  v10 = sub_1643280(*a1);
  if ( (unsigned __int8)sub_38A1070(a1, v10, &v47, a3, a4, a5, a6) )
    return 1;
  v41 = sub_388AF10((__int64)a1, 7, "expected ']' in phi value list");
  if ( v41 )
    return 1;
  v11 = 16;
  v50 = v52;
  v51 = 0x1000000000LL;
  v12 = 0;
  while ( 1 )
  {
    v13 = v47;
    v14 = v46;
    if ( (unsigned int)v12 >= v11 )
    {
      v39 = v46;
      v40 = v47;
      sub_16CD150((__int64)&v50, v52, 0, 16, v47, v46);
      v12 = (unsigned int)v51;
      v14 = v39;
      v13 = v40;
    }
    v15 = (char *)&v50[16 * v12];
    *(_QWORD *)v15 = v14;
    *((_QWORD *)v15 + 1) = v13;
    LODWORD(v51) = v51 + 1;
    if ( *((_DWORD *)a1 + 16) != 4 )
      break;
    v37 = sub_3887100((__int64)(a1 + 1));
    *((_DWORD *)a1 + 16) = v37;
    if ( v37 == 376 )
    {
      v41 = 1;
      break;
    }
    if ( (unsigned __int8)sub_388AF10((__int64)a1, 6, "expected '[' in phi value list")
      || (unsigned __int8)sub_38A1070(a1, v45, &v46, a3, a4, a5, a6)
      || (unsigned __int8)sub_388AF10((__int64)a1, 4, "expected ',' after insertelement value")
      || (v38 = sub_1643280(*a1), (unsigned __int8)sub_38A1070(a1, v38, &v47, a3, a4, a5, a6))
      || (unsigned __int8)sub_388AF10((__int64)a1, 7, "expected ']' in phi value list") )
    {
      result = 1;
      goto LABEL_36;
    }
    v12 = (unsigned int)v51;
    v11 = HIDWORD(v51);
  }
  v16 = v45;
  v42 = v45;
  if ( !*(_BYTE *)(v45 + 8) || *(_BYTE *)(v45 + 8) == 12 )
  {
    v48[0] = (__int64)"phi node must have first class type";
    v49 = 259;
    result = (unsigned __int8)sub_38814C0((__int64)(a1 + 1), v8, (__int64)v48);
  }
  else
  {
    v17 = v51;
    v49 = 257;
    v18 = sub_1648B60(64);
    v22 = v18;
    if ( v18 )
    {
      sub_15F1EA0(v18, v42, 53, 0, 0, 0);
      *(_DWORD *)(v22 + 56) = v17;
      sub_164B780(v22, v48);
      v16 = *(unsigned int *)(v22 + 56);
      sub_1648880(v22, v16, 1);
    }
    v23 = 0;
    v24 = 16LL * (unsigned int)v51;
    if ( (_DWORD)v51 )
    {
      v43 = a2;
      v25 = 16LL * (unsigned int)v51;
      do
      {
        v32 = *(_QWORD *)&v50[v23 + 8];
        v33 = *(_QWORD *)&v50[v23];
        v34 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
        if ( v34 == *(_DWORD *)(v22 + 56) )
        {
          sub_15F55D0(v22, v16, v19, v24, v20, v21);
          v34 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
        }
        v35 = (v34 + 1) & 0xFFFFFFF;
        v16 = (unsigned int)(v35 - 1);
        v36 = v35 | *(_DWORD *)(v22 + 20) & 0xF0000000;
        *(_DWORD *)(v22 + 20) = v36;
        if ( (v36 & 0x40000000) != 0 )
          v26 = *(_QWORD *)(v22 - 8);
        else
          v26 = v22 - 24 * v35;
        v27 = (_QWORD *)(v26 + 24LL * (unsigned int)v16);
        if ( *v27 )
        {
          v16 = v27[1];
          v28 = v27[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v28 = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v28;
        }
        *v27 = v33;
        if ( v33 )
        {
          v29 = *(_QWORD *)(v33 + 8);
          v27[1] = v29;
          if ( v29 )
          {
            v20 = (__int64)(v27 + 1);
            v16 = (unsigned __int64)(v27 + 1) | *(_QWORD *)(v29 + 16) & 3LL;
            *(_QWORD *)(v29 + 16) = v16;
          }
          v27[2] = (v33 + 8) | v27[2] & 3LL;
          *(_QWORD *)(v33 + 8) = v27;
        }
        v30 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
        v31 = (unsigned int)(v30 - 1);
        if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
          v24 = *(_QWORD *)(v22 - 8);
        else
          v24 = v22 - 24 * v30;
        v23 += 16;
        v19 = 3LL * *(unsigned int *)(v22 + 56);
        *(_QWORD *)(v24 + 8 * v31 + 24LL * *(unsigned int *)(v22 + 56) + 8) = v32;
      }
      while ( v25 != v23 );
      a2 = v43;
    }
    *a2 = v22;
    result = 2 * (unsigned int)(v41 != 0);
  }
LABEL_36:
  if ( v50 != v52 )
  {
    v44 = result;
    _libc_free((unsigned __int64)v50);
    return v44;
  }
  return result;
}
