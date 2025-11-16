// Function: sub_15D57B0
// Address: 0x15d57b0
//
_QWORD *__fastcall sub_15D57B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  __int64 v4; // rax
  int v5; // ebx
  __int64 v6; // r14
  __int64 v7; // rsi
  int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned int v14; // eax
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 v17; // r13
  int v18; // r8d
  __int64 v19; // rcx
  unsigned int v20; // eax
  __int64 *v21; // r12
  __int64 v22; // rsi
  unsigned __int64 v23; // rdi
  unsigned int v24; // ebx
  __int64 v25; // rax
  unsigned int v26; // r11d
  __int64 *v27; // r13
  int v28; // r14d
  _QWORD *v29; // r10
  unsigned int v30; // r8d
  __int64 v31; // r14
  _QWORD *v32; // rsi
  int v33; // r8d
  unsigned int v34; // r9d
  __int64 v35; // rdx
  __int64 *v36; // rax
  unsigned int v37; // r11d
  _QWORD *v38; // rbx
  _QWORD *v39; // r12
  unsigned __int64 v40; // rdi
  _QWORD *v41; // rbx
  _QWORD *v42; // r12
  unsigned __int64 v43; // rdi
  unsigned int v46; // [rsp+10h] [rbp-150h]
  unsigned int v48; // [rsp+20h] [rbp-140h]
  int v50; // [rsp+30h] [rbp-130h]
  __int64 v51; // [rsp+30h] [rbp-130h]
  __int64 v52; // [rsp+38h] [rbp-128h]
  __int64 v53; // [rsp+40h] [rbp-120h]
  __int64 v54; // [rsp+58h] [rbp-108h] BYREF
  __int64 v55; // [rsp+60h] [rbp-100h] BYREF
  __int64 v56; // [rsp+68h] [rbp-F8h]
  __int64 v57; // [rsp+70h] [rbp-F0h]
  char v58[8]; // [rsp+78h] [rbp-E8h] BYREF
  _QWORD *v59; // [rsp+80h] [rbp-E0h]
  int v60; // [rsp+88h] [rbp-D8h]
  int v61; // [rsp+8Ch] [rbp-D4h]
  unsigned int v62; // [rsp+90h] [rbp-D0h]
  __int64 v63[4]; // [rsp+A0h] [rbp-C0h] BYREF
  _QWORD *v64; // [rsp+C0h] [rbp-A0h]
  unsigned int v65; // [rsp+D0h] [rbp-90h]
  unsigned __int64 *v66; // [rsp+E0h] [rbp-80h] BYREF
  _BYTE *v67; // [rsp+E8h] [rbp-78h]
  unsigned __int64 v68[2]; // [rsp+F0h] [rbp-70h] BYREF
  int v69; // [rsp+100h] [rbp-60h]
  _BYTE v70[88]; // [rsp+108h] [rbp-58h] BYREF

  v3 = a1;
  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  sub_15CDF00((__int64)&v55, a3);
  sub_15D45B0((__int64)&v55);
  v4 = *(_QWORD *)(a2 + 88);
  v53 = v4 + 72;
  if ( *(_QWORD *)(v4 + 80) == v4 + 72 )
    goto LABEL_49;
  v5 = 0;
  v48 = 1;
  v6 = *(_QWORD *)(v4 + 80);
  while ( 1 )
  {
    v7 = v6 - 24;
    if ( !v6 )
      v7 = 0;
    v63[0] = v7;
    sub_15CF8B0((__int64)&v66, v7, a3);
    v8 = (int)v67;
    if ( v66 != v68 )
    {
      v50 = (int)v67;
      _libc_free((unsigned __int64)v66);
      v8 = v50;
    }
    if ( !v8 )
      break;
    v6 = *(_QWORD *)(v6 + 8);
    if ( v53 == v6 )
      goto LABEL_11;
LABEL_4:
    ++v5;
  }
  sub_15CDD90((__int64)a1, v63);
  v9 = sub_15D5190((__int64)&v55, v63[0], v48, (unsigned __int8 (__fastcall *)(char *))sub_15CBC50, 1);
  v6 = *(_QWORD *)(v6 + 8);
  v48 = v9;
  if ( v53 != v6 )
    goto LABEL_4;
LABEL_11:
  v10 = v48;
  v3 = a1;
  if ( v48 != v5 + 2 )
  {
    v66 = 0;
    v67 = v70;
    v68[0] = (unsigned __int64)v70;
    v68[1] = 4;
    v11 = *(_QWORD *)(a2 + 88);
    v69 = 0;
    v12 = *(_QWORD *)(v11 + 80);
    v51 = v11 + 72;
    if ( v11 + 72 != v12 )
    {
      do
      {
        v13 = 0;
        if ( v12 )
          v13 = v12 - 24;
        v54 = v13;
        if ( !(unsigned __int8)sub_15CE6E0((__int64)v58, &v54, v63) )
        {
          v14 = sub_15D54A0((__int64)&v55, v13, v10, (unsigned __int8 (__fastcall *)(char *))sub_15CBC50, v10);
          v15 = v14;
          v46 = v14;
          v16 = 8LL * v14;
          v63[0] = *(_QWORD *)(v55 + v16);
          sub_1412190((__int64)&v66, v63[0]);
          sub_15CDD90((__int64)a1, v63);
          if ( v46 > v10 )
          {
            v17 = 8 * (v15 - (v46 + ~v10));
            while ( 1 )
            {
              if ( v62 )
              {
                v18 = 1;
                v19 = *(_QWORD *)(v55 + v16);
                v20 = (v62 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
                v21 = &v59[9 * v20];
                v22 = *v21;
                if ( v19 == *v21 )
                {
LABEL_21:
                  v23 = v21[5];
                  if ( (__int64 *)v23 != v21 + 7 )
                    _libc_free(v23);
                  *v21 = -16;
                  --v60;
                  ++v61;
                }
                else
                {
                  while ( v22 != -8 )
                  {
                    v20 = (v62 - 1) & (v18 + v20);
                    v21 = &v59[9 * v20];
                    v22 = *v21;
                    if ( v19 == *v21 )
                      goto LABEL_21;
                    ++v18;
                  }
                }
              }
              v56 -= 8;
              if ( v16 == v17 )
                break;
              v16 -= 8;
            }
          }
          v10 = sub_15D5190((__int64)&v55, v63[0], v10, (unsigned __int8 (__fastcall *)(char *))sub_15CBC50, 1);
        }
        v12 = *(_QWORD *)(v12 + 8);
      }
      while ( v51 != v12 );
      v3 = a1;
      if ( (_BYTE *)v68[0] != v67 )
        _libc_free(v68[0]);
    }
    v24 = 0;
    sub_15CDF00((__int64)v63, a3);
    v25 = 0;
    if ( *((_DWORD *)v3 + 2) )
    {
      do
      {
        while ( 1 )
        {
          v27 = (__int64 *)(*v3 + 8 * v25);
          sub_15CF8B0((__int64)&v66, *v27, a3);
          v28 = (int)v67;
          if ( v66 != v68 )
            _libc_free((unsigned __int64)v66);
          if ( !v28 )
            break;
          sub_15CED80((__int64)v63);
          if ( (unsigned int)sub_15D54A0((__int64)v63, *v27, 0, (unsigned __int8 (__fastcall *)(char *))sub_15CBC50, 0) <= 1 )
            break;
          v29 = (_QWORD *)*v3;
          v30 = 2;
          v31 = v63[0];
          v52 = *((unsigned int *)v3 + 2);
          v32 = (_QWORD *)(*v3 + v52 * 8);
          while ( 1 )
          {
            v66 = *(unsigned __int64 **)(v31 + 8LL * v30);
            if ( v32 != sub_15CBCA0(v29, (__int64)v32, (__int64 *)&v66) )
              break;
            v30 = v33 + 1;
            if ( v34 < v30 )
              goto LABEL_32;
          }
          v35 = *v27;
          v36 = &v29[v52 - 1];
          *v27 = *v36;
          *v36 = v35;
          v37 = *((_DWORD *)v3 + 2) - 1;
          v25 = v24;
          *((_DWORD *)v3 + 2) = v37;
          if ( v37 <= v24 )
            goto LABEL_41;
        }
        v26 = *((_DWORD *)v3 + 2);
LABEL_32:
        v25 = ++v24;
      }
      while ( v26 > v24 );
    }
LABEL_41:
    if ( v65 )
    {
      v38 = v64;
      v39 = &v64[9 * v65];
      do
      {
        if ( *v38 != -8 && *v38 != -16 )
        {
          v40 = v38[5];
          if ( (_QWORD *)v40 != v38 + 7 )
            _libc_free(v40);
        }
        v38 += 9;
      }
      while ( v39 != v38 );
    }
    j___libc_free_0(v64);
    sub_15CE080(v63);
  }
LABEL_49:
  if ( v62 )
  {
    v41 = v59;
    v42 = &v59[9 * v62];
    do
    {
      if ( *v41 != -16 && *v41 != -8 )
      {
        v43 = v41[5];
        if ( (_QWORD *)v43 != v41 + 7 )
          _libc_free(v43);
      }
      v41 += 9;
    }
    while ( v42 != v41 );
  }
  j___libc_free_0(v59);
  if ( v55 )
    j_j___libc_free_0(v55, v57 - v55);
  return v3;
}
