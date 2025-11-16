// Function: sub_2B51430
// Address: 0x2b51430
//
__int64 __fastcall sub_2B51430(__int64 a1, unsigned __int8 **a2, unsigned __int64 a3, unsigned int a4, __int64 a5)
{
  unsigned __int8 **v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // r8
  __int64 v11; // r9
  int v12; // edx
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  int v15; // edx
  __int64 ***v16; // rax
  __int64 **v17; // r15
  unsigned __int8 *v18; // rsi
  int v19; // eax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // edx
  int v27; // eax
  unsigned int v28; // r13d
  unsigned __int64 v29; // rax
  unsigned __int8 **v30; // r12
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int8 *v33; // r14
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // r12d
  unsigned int v39; // r13d
  __int64 v40; // rbx
  unsigned int v41; // edx
  __int64 v42; // rcx
  unsigned __int8 **v43; // r13
  __int64 v44; // r15
  __int64 v45; // rbx
  __int64 v46; // rdi
  int v47; // eax
  __int64 v48; // rdx
  __int64 *v49; // rax
  __int64 *v50; // rdi
  __int64 v51; // r12
  unsigned __int64 v52; // r12
  unsigned __int64 v53; // r13
  unsigned __int64 v54; // rbx
  __int64 v55; // rax
  __int64 *v56; // rdx
  __int64 *v57; // rcx
  __int64 v58; // rsi
  __int64 v59; // r12
  __int64 *v60; // rsi
  __int64 v61; // rbx
  int v62; // edx
  bool v63; // cc
  unsigned __int8 **v64; // [rsp+18h] [rbp-B8h]
  __int64 v65; // [rsp+18h] [rbp-B8h]
  __int64 *v66; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v67; // [rsp+28h] [rbp-A8h]
  _BYTE v68[48]; // [rsp+30h] [rbp-A0h] BYREF
  void *s; // [rsp+60h] [rbp-70h] BYREF
  __int64 v70; // [rsp+68h] [rbp-68h]
  _BYTE v71[96]; // [rsp+70h] [rbp-60h] BYREF

  v8 = a2;
  v9 = sub_2B50F20(a1, (__int64)a2, a3, a5);
  if ( v12 == 1 )
    *(_DWORD *)(a1 + 128) = 1;
  if ( __OFADD__(*(_QWORD *)(a1 + 120), v9) )
  {
    v63 = v9 <= 0;
    v13 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v63 )
      v13 = 0x8000000000000000LL;
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 120) + v9;
  }
  *(_QWORD *)(a1 + 120) = v13;
  if ( a5 )
  {
    v14 = *(_QWORD *)a1;
    v15 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)(v15 - 17) > 1
      || (v14 = **(_QWORD **)(v14 + 16), v15 = *(unsigned __int8 *)(v14 + 8), (unsigned int)(v15 - 17) > 1) )
    {
      if ( (_BYTE)v15 != 14 )
        goto LABEL_9;
    }
    else
    {
      v16 = *(__int64 ****)(v14 + 16);
      v17 = *v16;
      if ( *((_BYTE *)*v16 + 8) != 14 )
      {
LABEL_9:
        v18 = (unsigned __int8 *)sub_AD62B0(v14);
LABEL_10:
        v19 = *(_DWORD *)(*(_QWORD *)(a5 + 8) + 32LL);
        BYTE4(v66) = 0;
        LODWORD(v66) = v19;
        return sub_AD5E10((__int64)v66, v18);
      }
      if ( (unsigned __int8)(v15 - 17) <= 1u )
      {
LABEL_13:
        v21 = sub_9208B0(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 3344LL), (__int64)v17);
        v70 = v22;
        s = (void *)((v21 + 7) & 0xFFFFFFFFFFFFFFF8LL);
        v23 = sub_CA1930(&s);
        v24 = sub_BCCE00(*(_QWORD **)v14, v23);
        v25 = sub_AD62B0(v24);
        v18 = (unsigned __int8 *)sub_AD4C70(v25, v17, 0);
        v26 = *(unsigned __int8 *)(v14 + 8);
        if ( (unsigned int)(v26 - 17) <= 1 )
        {
          v27 = *(_DWORD *)(v14 + 32);
          BYTE4(s) = (_BYTE)v26 == 18;
          LODWORD(s) = v27;
          v18 = (unsigned __int8 *)sub_AD5E10((__int64)s, v18);
        }
        goto LABEL_10;
      }
    }
    v17 = (__int64 **)v14;
    goto LABEL_13;
  }
  v28 = a3;
  v66 = (__int64 *)v68;
  v67 = 0x600000000LL;
  if ( a4 )
  {
    if ( (unsigned int)a3 <= a4 )
      a4 = a3;
    v28 = a4;
  }
  v29 = v28;
  if ( v28 > a3 )
    v29 = a3;
  v30 = &a2[v29];
  if ( v30 == a2 )
  {
    v36 = (__int64 *)v68;
  }
  else
  {
    do
    {
      while ( 1 )
      {
        v33 = *v8;
        if ( (unsigned int)**v8 - 12 <= 1 )
          break;
        v34 = sub_AD6530(*((_QWORD *)v33 + 1), (__int64)a2);
        v35 = (unsigned int)v67;
        v10 = (unsigned int)v67 + 1LL;
        if ( v10 > HIDWORD(v67) )
        {
          a2 = (unsigned __int8 **)v68;
          v65 = v34;
          sub_C8D5F0((__int64)&v66, v68, (unsigned int)v67 + 1LL, 8u, v10, v11);
          v35 = (unsigned int)v67;
          v34 = v65;
        }
        ++v8;
        v66[v35] = v34;
        LODWORD(v67) = v67 + 1;
        if ( v30 == v8 )
          goto LABEL_27;
      }
      v31 = (unsigned int)v67;
      v32 = (unsigned int)v67 + 1LL;
      if ( v32 > HIDWORD(v67) )
      {
        a2 = (unsigned __int8 **)v68;
        sub_C8D5F0((__int64)&v66, v68, v32, 8u, v10, v11);
        v31 = (unsigned int)v67;
      }
      ++v8;
      v66[v31] = (__int64)v33;
      LODWORD(v67) = v67 + 1;
    }
    while ( v30 != v8 );
LABEL_27:
    v36 = v66;
  }
  v37 = *(_QWORD *)(*v36 + 8);
  if ( *(_BYTE *)(v37 + 8) == 17 )
  {
    v38 = *(_DWORD *)(v37 + 32);
    v39 = v38 * v28;
    s = v71;
    v70 = 0x600000000LL;
    if ( v39 > 6 )
    {
      sub_C8D5F0((__int64)&s, v71, v39, 8u, v10, v11);
      a2 = 0;
      memset(s, 0, 8LL * v39);
      LODWORD(v70) = v39;
      v36 = v66;
    }
    else
    {
      if ( v39 )
      {
        v40 = 8LL * v39;
        if ( v40 )
        {
          a2 = 0;
          *(_QWORD *)&v71[(unsigned int)v40 - 8] = 0;
          if ( (unsigned int)(v40 - 1) >= 8 )
          {
            v41 = 0;
            do
            {
              v42 = v41;
              v41 += 8;
              *(_QWORD *)&v71[v42] = 0;
            }
            while ( v41 < (((_DWORD)v40 - 1) & 0xFFFFFFF8) );
          }
        }
      }
      LODWORD(v70) = v39;
    }
    v64 = (unsigned __int8 **)&v36[(unsigned int)v67];
    if ( v64 != (unsigned __int8 **)v36 )
    {
      v43 = (unsigned __int8 **)v36;
      v44 = 0;
      v45 = 8LL * v38;
      do
      {
        v46 = *((_QWORD *)*v43 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v46 + 8) - 17 <= 1 )
          v46 = **(_QWORD **)(v46 + 16);
        v47 = **v43;
        if ( (_BYTE)v47 == 13 )
        {
          v48 = sub_ACADE0((__int64 **)v46);
        }
        else if ( (unsigned int)(v47 - 12) > 1 )
        {
          v48 = sub_AD6530(v46, (__int64)a2);
        }
        else
        {
          v48 = sub_ACA8A0((__int64 **)v46);
        }
        if ( v38 )
        {
          v49 = (__int64 *)((char *)s + v44);
          v50 = (__int64 *)((char *)s + v44 + v45);
          do
            *v49++ = v48;
          while ( v50 != v49 );
        }
        ++v43;
        v44 += v45;
      }
      while ( v64 != v43 );
      v36 = v66;
    }
    if ( v36 == (__int64 *)v68 || s == v71 )
    {
      if ( HIDWORD(v67) < (unsigned int)v70 )
        sub_C8D5F0((__int64)&v66, v68, (unsigned int)v70, 8u, v10, v11);
      v52 = (unsigned int)v67;
      if ( (unsigned int)v67 > (unsigned __int64)HIDWORD(v70) )
      {
        sub_C8D5F0((__int64)&s, v71, (unsigned int)v67, 8u, v10, v11);
        v52 = (unsigned int)v67;
      }
      v53 = (unsigned int)v70;
      v54 = v52;
      if ( (unsigned int)v70 <= v52 )
        v54 = (unsigned int)v70;
      if ( v54 )
      {
        v55 = 0;
        do
        {
          v56 = (__int64 *)((char *)s + v55 * 8);
          v57 = &v66[v55++];
          v58 = *v57;
          *v57 = *v56;
          *v56 = v58;
        }
        while ( v54 != v55 );
        v52 = (unsigned int)v67;
        v53 = (unsigned int)v70;
      }
      v36 = (__int64 *)s;
      if ( v52 <= v53 )
      {
        if ( v52 < v53 )
        {
          v61 = 8 * v54;
          v62 = v52;
          if ( (char *)s + v61 != (char *)s + 8 * v53 )
          {
            memcpy(&v66[v52], (char *)s + v61, 8 * v53 - v61);
            v62 = v67;
            v36 = (__int64 *)s;
          }
          LODWORD(v67) = v53 - v52 + v62;
        }
      }
      else
      {
        v59 = v52;
        v60 = &v66[v54];
        if ( v60 != &v66[v59] )
        {
          memcpy((char *)s + 8 * v53, v60, v59 * 8 - 8 * v54);
          v36 = (__int64 *)s;
        }
        LODWORD(v67) = v54;
      }
    }
    else
    {
      v67 = v70;
      v66 = (__int64 *)s;
    }
    if ( v36 != (__int64 *)v71 )
      _libc_free((unsigned __int64)v36);
    v36 = v66;
  }
  v51 = sub_AD3730(v36, (unsigned int)v67);
  if ( v66 != (__int64 *)v68 )
    _libc_free((unsigned __int64)v66);
  return v51;
}
