// Function: sub_38538A0
// Address: 0x38538a0
//
bool __fastcall sub_38538A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 ***v13; // r12
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  unsigned int v17; // esi
  __int64 ****v18; // rax
  __int64 ***v19; // rdi
  __int64 ***v20; // r8
  __int64 **v21; // r8
  __int64 v22; // r15
  __int64 ****v23; // rbx
  __int64 v24; // r15
  __int64 v25; // r15
  unsigned __int64 v26; // r9
  unsigned __int64 v27; // r12
  __int64 ****v28; // rdx
  int v29; // ecx
  __int64 ****v30; // r10
  __int64 ****v31; // rax
  __int64 ***v32; // rcx
  int v33; // ebx
  bool result; // al
  int v35; // eax
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rax
  __int64 ****v39; // rax
  __int64 **v40; // [rsp+0h] [rbp-80h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+8h] [rbp-78h]
  __int64 v44; // [rsp+8h] [rbp-78h]
  __int64 v45; // [rsp+18h] [rbp-68h] BYREF
  __int64 ****v46; // [rsp+20h] [rbp-60h] BYREF
  __int64 v47; // [rsp+28h] [rbp-58h]
  _BYTE v48[80]; // [rsp+30h] [rbp-50h] BYREF

  v47 = 0x200000000LL;
  v8 = *(_DWORD *)(a2 + 20);
  v46 = (__int64 ****)v48;
  v9 = 24LL * (v8 & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v10 = *(_QWORD *)(a2 - 8);
    v11 = v10 + v9;
  }
  else
  {
    v10 = a2 - v9;
    v11 = a2;
  }
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v13 = *(__int64 ****)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 16LL) > 0x10u )
      {
        v14 = *(_DWORD *)(a1 + 160);
        if ( !v14 )
          goto LABEL_12;
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 144);
        v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = (__int64 ****)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v13 != *v18 )
        {
          v37 = 1;
          while ( v19 != (__int64 ***)-8LL )
          {
            a6 = v37 + 1;
            v38 = v15 & (v17 + v37);
            v17 = v38;
            v18 = (__int64 ****)(v16 + 16 * v38);
            v19 = *v18;
            if ( v13 == *v18 )
              goto LABEL_11;
            v37 = a6;
          }
          goto LABEL_12;
        }
LABEL_11:
        v13 = v18[1];
        if ( !v13 )
          goto LABEL_12;
      }
      v12 = (unsigned int)v47;
      if ( (unsigned int)v47 >= HIDWORD(v47) )
      {
        v43 = a1;
        sub_16CD150((__int64)&v46, v48, 0, 8, a1, a6);
        v12 = (unsigned int)v47;
        a1 = v43;
      }
      v10 += 24;
      v46[v12] = v13;
      LODWORD(v47) = v47 + 1;
      if ( v11 == v10 )
      {
        v39 = v46;
        goto LABEL_32;
      }
    }
  }
  v39 = (__int64 ****)v48;
LABEL_32:
  v44 = a1;
  v36 = sub_15A46C0((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24, *v39, *(__int64 ***)a2, 0);
  a1 = v44;
  if ( v36 )
  {
    v45 = a2;
    sub_38526A0(v44 + 136, &v45)[1] = v36;
    result = 1;
    if ( v46 != (__int64 ****)v48 )
    {
      _libc_free((unsigned __int64)v46);
      return 1;
    }
  }
  else
  {
LABEL_12:
    if ( v46 != (__int64 ****)v48 )
    {
      v41 = a1;
      _libc_free((unsigned __int64)v46);
      a1 = v41;
    }
    v42 = a1;
    sub_384F350(a1, *(_QWORD *)(a2 - 24));
    v20 = (__int64 ***)v42;
    if ( (unsigned int)*(unsigned __int8 *)(a2 + 16) - 63 <= 5 )
    {
      v35 = sub_14A3050(*(_QWORD *)v42);
      v20 = (__int64 ***)v42;
      if ( v35 == 4 )
        *(_DWORD *)(v42 + 76) += 25;
    }
    v21 = *v20;
    v22 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v23 = *(__int64 *****)(a2 - 8);
      v24 = (__int64)&v23[v22];
    }
    else
    {
      v23 = (__int64 ****)(a2 - v22 * 8);
      v24 = a2;
    }
    v25 = v24 - (_QWORD)v23;
    v46 = (__int64 ****)v48;
    v47 = 0x400000000LL;
    v26 = 0xAAAAAAAAAAAAAAABLL * (v25 >> 3);
    v27 = v26;
    if ( (unsigned __int64)v25 > 0x60 )
    {
      v40 = v21;
      sub_16CD150((__int64)&v46, v48, 0xAAAAAAAAAAAAAAABLL * (v25 >> 3), 8, (int)v21, v26);
      v30 = v46;
      v29 = v47;
      v26 = 0xAAAAAAAAAAAAAAABLL * (v25 >> 3);
      v21 = v40;
      v28 = &v46[(unsigned int)v47];
    }
    else
    {
      v28 = (__int64 ****)v48;
      v29 = 0;
      v30 = (__int64 ****)v48;
    }
    if ( v25 > 0 )
    {
      v31 = v23;
      do
      {
        v32 = *v31;
        ++v28;
        v31 += 3;
        *(v28 - 1) = v32;
        --v27;
      }
      while ( v27 );
      v30 = v46;
      v29 = v47;
    }
    LODWORD(v47) = v26 + v29;
    v33 = sub_14A5330(v21, a2, (__int64)v30, (unsigned int)(v26 + v29));
    if ( v46 != (__int64 ****)v48 )
      _libc_free((unsigned __int64)v46);
    return v33 == 0;
  }
  return result;
}
