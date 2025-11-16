// Function: sub_38139C0
// Address: 0x38139c0
//
unsigned __int8 *__fastcall sub_38139C0(__int128 a1, __int64 a2, unsigned int a3, char a4, _QWORD *a5, __m128i a6)
{
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  unsigned int v13; // r11d
  unsigned __int64 v14; // rax
  __int128 v15; // rax
  __int64 v16; // r9
  unsigned __int8 *v17; // rax
  unsigned int v18; // r11d
  unsigned __int8 *v19; // r12
  unsigned int v20; // edx
  unsigned __int64 v21; // r13
  unsigned int v22; // edx
  unsigned int v23; // esi
  __int128 v24; // rax
  __int64 v25; // r9
  unsigned __int8 *v26; // r12
  unsigned __int8 *result; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int128 v31; // rax
  __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // r8
  __int128 v35; // [rsp-30h] [rbp-D0h]
  unsigned int v36; // [rsp+0h] [rbp-A0h]
  __int64 v37; // [rsp+0h] [rbp-A0h]
  unsigned int v38; // [rsp+0h] [rbp-A0h]
  unsigned int v39; // [rsp+0h] [rbp-A0h]
  unsigned int v40; // [rsp+0h] [rbp-A0h]
  unsigned int v42; // [rsp+Ch] [rbp-94h]
  unsigned __int8 *v44; // [rsp+10h] [rbp-90h]
  unsigned int v45; // [rsp+40h] [rbp-60h] BYREF
  __int64 v46; // [rsp+48h] [rbp-58h]
  unsigned __int64 v47; // [rsp+50h] [rbp-50h] BYREF
  __int64 v48; // [rsp+58h] [rbp-48h]
  __int64 v49; // [rsp+60h] [rbp-40h]
  __int64 v50; // [rsp+68h] [rbp-38h]

  v8 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * DWORD2(a1));
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v45) = v9;
  v46 = v10;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0xD3u )
    {
      LOWORD(v47) = v9;
      v48 = v10;
LABEL_4:
      v11 = (unsigned __int16)v9;
      if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
        BUG();
      v12 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v9 - 16];
      v13 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
      if ( a4 )
        goto LABEL_7;
      goto LABEL_30;
    }
    LOWORD(v9) = word_4456580[v9 - 1];
    v28 = 0;
  }
  else
  {
    v37 = v10;
    if ( !sub_30070B0((__int64)&v45) )
    {
      v48 = v37;
      LOWORD(v47) = 0;
      goto LABEL_29;
    }
    LOWORD(v9) = sub_3009970((__int64)&v45, *((__int64 *)&a1 + 1), v37, v33, v34);
  }
  LOWORD(v47) = v9;
  v48 = v28;
  if ( (_WORD)v9 )
    goto LABEL_4;
LABEL_29:
  v12 = sub_3007260((__int64)&v47);
  v49 = v12;
  v13 = v12;
  v50 = v29;
  if ( a4 )
  {
LABEL_7:
    LODWORD(v48) = v12;
    v42 = a3 - 1;
    if ( (unsigned int)v12 > 0x40 )
    {
      v39 = v12;
      sub_C43690((__int64)&v47, 0, 0);
      v13 = v39;
    }
    else
    {
      v47 = 0;
    }
    if ( a3 != 1 )
    {
      if ( v42 > 0x40 )
      {
        v40 = v13;
        sub_C43C90(&v47, 0, v42);
        v13 = v40;
      }
      else
      {
        v14 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)a3);
        if ( (unsigned int)v48 > 0x40 )
          *(_QWORD *)v47 |= v14;
        else
          v47 |= v14;
      }
    }
    v36 = v13;
    *(_QWORD *)&v15 = sub_34007B0((__int64)a5, (__int64)&v47, a2, v45, v46, 0, a6, 0);
    v17 = sub_3406EB0(a5, 0xB4u, a2, v45, v46, v16, a1, v15);
    v18 = v36;
    v19 = v17;
    v21 = v20 | *((_QWORD *)&a1 + 1) & 0xFFFFFFFF00000000LL;
    if ( (unsigned int)v48 > 0x40 && v47 )
    {
      j_j___libc_free_0_0(v47);
      v18 = v36;
    }
    LODWORD(v48) = v18;
    if ( v18 > 0x40 )
    {
      v38 = v18;
      sub_C43690((__int64)&v47, 0, 0);
      v22 = v48;
      v18 = v38;
    }
    else
    {
      v47 = 0;
      v22 = v18;
    }
    v23 = v22 + v42 - v18;
    if ( v22 != v23 )
    {
      if ( v23 > 0x3F || v22 > 0x40 )
        sub_C43C90(&v47, v23, v22);
      else
        v47 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)a3 + 63 - (unsigned __int8)v18) << v23;
    }
    *(_QWORD *)&v24 = sub_34007B0((__int64)a5, (__int64)&v47, a2, v45, v46, 0, a6, 0);
    *((_QWORD *)&v35 + 1) = v21;
    *(_QWORD *)&v35 = v19;
    v26 = sub_3406EB0(a5, 0xB5u, a2, v45, v46, v25, v35, v24);
    if ( (unsigned int)v48 > 0x40 )
    {
      if ( v47 )
        j_j___libc_free_0_0(v47);
    }
    return v26;
  }
LABEL_30:
  LODWORD(v48) = v12;
  if ( (unsigned int)v12 > 0x40 )
    sub_C43690((__int64)&v47, 0, 0);
  else
    v47 = 0;
  if ( a3 )
  {
    if ( a3 > 0x40 )
    {
      sub_C43C90(&v47, 0, a3);
    }
    else
    {
      v30 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a3);
      if ( (unsigned int)v48 > 0x40 )
        *(_QWORD *)v47 |= v30;
      else
        v47 |= v30;
    }
  }
  *(_QWORD *)&v31 = sub_34007B0((__int64)a5, (__int64)&v47, a2, v45, v46, 0, a6, 0);
  result = sub_3406EB0(a5, 0xB6u, a2, v45, v46, v32, a1, v31);
  if ( (unsigned int)v48 > 0x40 && v47 )
  {
    v44 = result;
    j_j___libc_free_0_0(v47);
    return v44;
  }
  return result;
}
