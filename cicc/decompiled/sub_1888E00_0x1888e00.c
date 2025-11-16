// Function: sub_1888E00
// Address: 0x1888e00
//
unsigned __int64 __fastcall sub_1888E00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // rcx
  __int64 v15; // r13
  signed __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 result; // rax
  __int64 v19; // r15
  unsigned __int64 v20; // r13
  __int64 j; // r12
  __int64 v22; // rbx
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // r12
  __int64 v26; // r15
  __int64 i; // r13
  __int64 v28; // rbx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // rdx
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rbx
  __int64 v39; // r15
  __int64 k; // r12
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  signed __int64 v46; // [rsp+10h] [rbp-60h]
  signed __int64 v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  signed __int64 v49; // [rsp+18h] [rbp-58h]
  __int64 v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+18h] [rbp-58h]
  signed __int64 v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  int v57; // [rsp+20h] [rbp-50h]
  __int64 v58; // [rsp+28h] [rbp-48h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  __int64 v61; // [rsp+30h] [rbp-40h]
  __int64 v62; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v61 = a1;
    v62 = a3;
    v8 = a7;
    if ( a5 <= a7 )
      v8 = a5;
    if ( v8 >= a4 )
    {
      v25 = a2;
      result = sub_1888CD0(a1, a2, a6);
      v26 = result;
      for ( i = a1 + 8; v26 != v7; i += 48 )
      {
        if ( v25 == v62 )
          return sub_1888CD0(v7, v26, i - 8);
        v28 = *(_QWORD *)(i + 8);
        result = *(_QWORD *)(v7 + 40);
        if ( *(_QWORD *)(v25 + 40) >= result )
        {
          for ( ; v28; result = j_j___libc_free_0(v31, 40) )
          {
            sub_1876060(*(_QWORD *)(v28 + 24));
            v31 = v28;
            v28 = *(_QWORD *)(v28 + 16);
          }
          *(_QWORD *)(i + 8) = 0;
          *(_QWORD *)(i + 16) = i;
          *(_QWORD *)(i + 24) = i;
          *(_QWORD *)(i + 32) = 0;
          if ( *(_QWORD *)(v7 + 16) )
          {
            *(_DWORD *)i = *(_DWORD *)(v7 + 8);
            v32 = *(_QWORD *)(v7 + 16);
            *(_QWORD *)(i + 8) = v32;
            *(_QWORD *)(i + 16) = *(_QWORD *)(v7 + 24);
            *(_QWORD *)(i + 24) = *(_QWORD *)(v7 + 32);
            *(_QWORD *)(v32 + 8) = i;
            *(_QWORD *)(i + 32) = *(_QWORD *)(v7 + 40);
            result = v7 + 8;
            *(_QWORD *)(v7 + 16) = 0;
            *(_QWORD *)(v7 + 24) = v7 + 8;
            *(_QWORD *)(v7 + 32) = v7 + 8;
            *(_QWORD *)(v7 + 40) = 0;
          }
          v7 += 48;
        }
        else
        {
          for ( ; v28; result = j_j___libc_free_0(v29, 40) )
          {
            sub_1876060(*(_QWORD *)(v28 + 24));
            v29 = v28;
            v28 = *(_QWORD *)(v28 + 16);
          }
          *(_QWORD *)(i + 8) = 0;
          *(_QWORD *)(i + 16) = i;
          *(_QWORD *)(i + 24) = i;
          *(_QWORD *)(i + 32) = 0;
          if ( *(_QWORD *)(v25 + 16) )
          {
            *(_DWORD *)i = *(_DWORD *)(v25 + 8);
            v30 = *(_QWORD *)(v25 + 16);
            *(_QWORD *)(i + 8) = v30;
            *(_QWORD *)(i + 16) = *(_QWORD *)(v25 + 24);
            *(_QWORD *)(i + 24) = *(_QWORD *)(v25 + 32);
            *(_QWORD *)(v30 + 8) = i;
            *(_QWORD *)(i + 32) = *(_QWORD *)(v25 + 40);
            result = v25 + 8;
            *(_QWORD *)(v25 + 16) = 0;
            *(_QWORD *)(v25 + 24) = v25 + 8;
            *(_QWORD *)(v25 + 32) = v25 + 8;
            *(_QWORD *)(v25 + 40) = 0;
          }
          v25 += 48;
        }
      }
      return result;
    }
    v9 = a5;
    if ( a5 <= a7 )
      break;
    if ( a5 >= a4 )
    {
      v55 = a4;
      v60 = a5 / 2;
      v15 = a2 + 16 * (a5 / 2 + ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
      v33 = sub_1873970(a1, a2, v15);
      v14 = v55;
      v16 = v60;
      v13 = v33;
      v10 = 0xAAAAAAAAAAAAAAABLL * ((v33 - a1) >> 4);
    }
    else
    {
      v58 = a4;
      v10 = a4 / 2;
      v11 = sub_1873900(a2, a3, a1 + 16 * (((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL) + a4 / 2));
      v14 = v58;
      v15 = v11;
      v16 = 0xAAAAAAAAAAAAAAABLL * ((v11 - a2) >> 4);
    }
    v59 = v14 - v10;
    if ( v14 - v10 <= v16 || v12 < v16 )
    {
      if ( v12 < v59 )
      {
        v48 = v12;
        v52 = v16;
        v57 = v13;
        v17 = sub_18887A0(v13, a2, v15);
        v12 = v48;
        v16 = v52;
        LODWORD(v13) = v57;
      }
      else
      {
        v17 = v15;
        if ( v59 )
        {
          v43 = v12;
          v46 = v16;
          v54 = v13;
          v50 = sub_1888CD0(v13, a2, v7);
          sub_1888CD0(a2, v15, v54);
          v17 = sub_1888BA0(v7, v50, v15);
          LODWORD(v13) = v54;
          v16 = v46;
          v12 = v43;
        }
      }
    }
    else
    {
      v17 = v13;
      if ( v16 )
      {
        v44 = v12;
        v47 = v16;
        v56 = v13;
        v51 = sub_1888CD0(a2, v15, v7);
        sub_1888BA0(v56, a2, v15);
        v17 = sub_1888CD0(v7, v51, v56);
        LODWORD(v13) = v56;
        v16 = v47;
        v12 = v44;
      }
    }
    v45 = v12;
    v49 = v16;
    v53 = v17;
    sub_1888E00(a1, v13, v17, v10, v16, v7, v12);
    a6 = v7;
    a2 = v15;
    a4 = v59;
    a7 = v45;
    a3 = v62;
    a5 = v9 - v49;
    a1 = v53;
  }
  result = sub_1888CD0(a2, a3, a6);
  if ( a1 == a2 )
    return sub_1888BA0(v7, result, v62);
  if ( v7 != result )
  {
    v19 = a2 - 48;
    v20 = result - 48;
    for ( j = v62 - 40; ; j -= 48 )
    {
      v22 = *(_QWORD *)(j + 8);
      result = *(_QWORD *)(v19 + 40);
      if ( *(_QWORD *)(v20 + 40) >= result )
      {
        for ( ; v22; result = j_j___libc_free_0(v34, 40) )
        {
          sub_1876060(*(_QWORD *)(v22 + 24));
          v34 = v22;
          v22 = *(_QWORD *)(v22 + 16);
        }
        *(_QWORD *)(j + 8) = 0;
        *(_QWORD *)(j + 16) = j;
        *(_QWORD *)(j + 24) = j;
        *(_QWORD *)(j + 32) = 0;
        if ( *(_QWORD *)(v20 + 16) )
        {
          *(_DWORD *)j = *(_DWORD *)(v20 + 8);
          v35 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(j + 8) = v35;
          *(_QWORD *)(j + 16) = *(_QWORD *)(v20 + 24);
          *(_QWORD *)(j + 24) = *(_QWORD *)(v20 + 32);
          *(_QWORD *)(v35 + 8) = j;
          *(_QWORD *)(j + 32) = *(_QWORD *)(v20 + 40);
          *(_QWORD *)(v20 + 16) = 0;
          *(_QWORD *)(v20 + 24) = v20 + 8;
          *(_QWORD *)(v20 + 32) = v20 + 8;
          *(_QWORD *)(v20 + 40) = 0;
        }
        if ( v7 == v20 )
          return result;
        v20 -= 48LL;
      }
      else
      {
        while ( v22 )
        {
          sub_1876060(*(_QWORD *)(v22 + 24));
          v23 = v22;
          v22 = *(_QWORD *)(v22 + 16);
          j_j___libc_free_0(v23, 40);
        }
        *(_QWORD *)(j + 8) = 0;
        *(_QWORD *)(j + 16) = j;
        *(_QWORD *)(j + 24) = j;
        *(_QWORD *)(j + 32) = 0;
        if ( *(_QWORD *)(v19 + 16) )
        {
          *(_DWORD *)j = *(_DWORD *)(v19 + 8);
          v24 = *(_QWORD *)(v19 + 16);
          *(_QWORD *)(j + 8) = v24;
          *(_QWORD *)(j + 16) = *(_QWORD *)(v19 + 24);
          *(_QWORD *)(j + 24) = *(_QWORD *)(v19 + 32);
          *(_QWORD *)(v24 + 8) = j;
          *(_QWORD *)(j + 32) = *(_QWORD *)(v19 + 40);
          *(_QWORD *)(v19 + 16) = 0;
          *(_QWORD *)(v19 + 24) = v19 + 8;
          *(_QWORD *)(v19 + 32) = v19 + 8;
          *(_QWORD *)(v19 + 40) = 0;
        }
        if ( v19 == v61 )
        {
          v36 = v20 + 48 - v7;
          result = 0xAAAAAAAAAAAAAAABLL * (v36 >> 4);
          v37 = result;
          if ( v36 > 0 )
          {
            v38 = v20 + 8;
            v39 = j - 48;
            do
            {
              for ( k = *(_QWORD *)(v39 + 8); k; result = j_j___libc_free_0(v41, 40) )
              {
                sub_1876060(*(_QWORD *)(k + 24));
                v41 = k;
                k = *(_QWORD *)(k + 16);
              }
              *(_QWORD *)(v39 + 8) = 0;
              *(_QWORD *)(v39 + 16) = v39;
              *(_QWORD *)(v39 + 24) = v39;
              *(_QWORD *)(v39 + 32) = 0;
              if ( *(_QWORD *)(v38 + 8) )
              {
                *(_DWORD *)v39 = *(_DWORD *)v38;
                v42 = *(_QWORD *)(v38 + 8);
                *(_QWORD *)(v39 + 8) = v42;
                *(_QWORD *)(v39 + 16) = *(_QWORD *)(v38 + 16);
                *(_QWORD *)(v39 + 24) = *(_QWORD *)(v38 + 24);
                *(_QWORD *)(v42 + 8) = v39;
                result = *(_QWORD *)(v38 + 32);
                *(_QWORD *)(v39 + 32) = result;
                *(_QWORD *)(v38 + 8) = 0;
                *(_QWORD *)(v38 + 16) = v38;
                *(_QWORD *)(v38 + 24) = v38;
                *(_QWORD *)(v38 + 32) = 0;
              }
              v39 -= 48;
              v38 -= 48LL;
              --v37;
            }
            while ( v37 );
          }
          return result;
        }
        v19 -= 48;
      }
    }
  }
  return result;
}
