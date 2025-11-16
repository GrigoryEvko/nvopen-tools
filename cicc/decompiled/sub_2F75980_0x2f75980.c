// Function: sub_2F75980
// Address: 0x2f75980
//
__m128i *__fastcall sub_2F75980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  int v8; // eax
  int v9; // eax
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // r8
  char v14; // al
  char v15; // di
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // r10
  __m128i *result; // rax
  _QWORD *v20; // r11
  __int64 v21; // r15
  __int64 v22; // r13
  __int64 v23; // r14
  unsigned int v24; // eax
  char v25; // di
  __int64 v26; // rax
  _QWORD *v27; // r10
  __int64 v28; // rsi
  __int64 v29; // r8
  __int64 v30; // rdx
  int v31; // eax
  __int64 v32; // r12
  _QWORD *v33; // r11
  __int64 v34; // rdx
  unsigned int v35; // eax
  __int16 *v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdx
  int v41; // eax
  __int64 v42; // r12
  __int16 *v43; // [rsp+8h] [rbp-88h]
  int v44; // [rsp+10h] [rbp-80h]
  __int16 *v45; // [rsp+10h] [rbp-80h]
  unsigned int v46; // [rsp+10h] [rbp-80h]
  int v47; // [rsp+10h] [rbp-80h]
  unsigned int v48; // [rsp+18h] [rbp-78h]
  __int64 v49; // [rsp+18h] [rbp-78h]
  unsigned int v50; // [rsp+18h] [rbp-78h]
  int v51; // [rsp+18h] [rbp-78h]
  unsigned int v52; // [rsp+18h] [rbp-78h]
  unsigned int v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  unsigned int v55; // [rsp+18h] [rbp-78h]
  __int64 v56; // [rsp+40h] [rbp-50h] BYREF
  __int64 v57; // [rsp+48h] [rbp-48h]
  _QWORD *v58; // [rsp+50h] [rbp-40h]
  char v59; // [rsp+58h] [rbp-38h]

  v6 = a1;
  v7 = a2;
  v8 = *(_DWORD *)(a2 + 44);
  v56 = a1;
  v57 = a3;
  v58 = (_QWORD *)a4;
  v9 = v8 & 4;
  v59 = a6;
  if ( !(_BYTE)a5 )
  {
    if ( v9 )
    {
      do
        v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v7 + 44) & 4) != 0 );
    }
    v21 = *(_QWORD *)(a2 + 24) + 48LL;
    while ( 1 )
    {
      v22 = *(_QWORD *)(v7 + 32);
      v23 = v22 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
      if ( v22 != v23 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v21 == v7 )
        break;
      if ( (*(_BYTE *)(v7 + 44) & 4) == 0 )
      {
        v7 = *(_QWORD *)(a2 + 24) + 48LL;
        break;
      }
    }
    if ( v23 == v22 )
    {
LABEL_50:
      v27 = *(_QWORD **)(v6 + 208);
      result = (__m128i *)(3LL * *(unsigned int *)(v6 + 216));
      if ( v27 != &v27[3 * *(unsigned int *)(v6 + 216)] )
      {
        while ( 1 )
        {
          result = sub_2F74650(v6 + 416, a2, a3, a4, a5, a6, *v27, v27[1], v27[2]);
          if ( v33 == v27 )
            break;
          v6 = v56;
        }
      }
      return result;
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v22 )
        goto LABEL_43;
      a4 = *(unsigned int *)(v22 + 8);
      if ( !(_DWORD)a4 )
        goto LABEL_43;
      a2 = *(unsigned __int8 *)(v22 + 4);
      v24 = *(unsigned __int8 *)(v22 + 3);
      v25 = *(_BYTE *)(v22 + 4) & 1;
      if ( (v24 & 0x10) == 0 )
      {
        if ( v25 )
          goto LABEL_43;
        a2 &= 2u;
        if ( (_DWORD)a2 )
          goto LABEL_43;
        if ( (int)a4 < 0 )
        {
          sub_2F747D0(v6, a2, a3, a4, a5, a6, *(_DWORD *)(v22 + 8), -1, -1);
        }
        else
        {
          a2 = (__int64)v58;
          v50 = *(_DWORD *)(v22 + 8);
          a5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v58 + 16LL) + 200LL))(*(_QWORD *)(*v58 + 16LL));
          a4 = v50;
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a5 + 248) + 16LL) + v50) )
          {
            v34 = 1LL << v50;
            a2 = v58[48];
            a4 = *(_QWORD *)(a2 + 8LL * (v50 >> 6)) & (1LL << v50);
            if ( !a4 )
            {
              a2 = *(_QWORD *)(v57 + 56);
              v35 = *(_DWORD *)(*(_QWORD *)(v57 + 8) + 24LL * v50 + 16);
              a4 = v35 & 0xFFF;
              v36 = (__int16 *)(a2 + 2LL * (v35 >> 12));
              do
              {
                v45 = v36;
                if ( !v36 )
                  break;
                v51 = a4;
                sub_2F747D0(v6, a2, v34, a4, a5, a6, a4, -1, -1);
                v36 = v45 + 1;
                a4 = (unsigned int)(*v45 + v51);
              }
              while ( *v45 );
            }
          }
        }
        goto LABEL_95;
      }
      if ( !v25 )
      {
        v28 = a2 & 2;
        if ( !(_DWORD)v28 && (*(_DWORD *)v22 & 0xFFF00) != 0 )
        {
          if ( (int)a4 < 0 )
          {
            v55 = *(_DWORD *)(v22 + 8);
            sub_2F747D0(v6, v28, a3, a4, a5, a6, v55, -1, -1);
            a4 = v55;
          }
          else
          {
            v52 = *(_DWORD *)(v22 + 8);
            a5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v58 + 16LL) + 200LL))(*(_QWORD *)(*v58 + 16LL));
            a4 = v52;
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a5 + 248) + 16LL) + v52) )
            {
              a5 = v58[48];
              if ( (*(_QWORD *)(a5 + 8LL * (v52 >> 6)) & (1LL << v52)) == 0 )
              {
                v37 = *(_DWORD *)(*(_QWORD *)(v57 + 8) + 24LL * v52 + 16);
                v38 = v37 & 0xFFF;
                v39 = *(_QWORD *)(v57 + 56) + 2LL * (v37 >> 12);
                do
                {
                  v43 = (__int16 *)v39;
                  if ( !v39 )
                    break;
                  v46 = a4;
                  sub_2F747D0(v6, v38, a3, a4, a5, a6, v38, -1, -1);
                  a4 = v46;
                  a5 = (unsigned int)*v43;
                  v39 = (__int64)(v43 + 1);
                  v38 = (unsigned int)(a5 + v38);
                }
                while ( *v43 );
              }
            }
          }
          v6 = v56;
          v24 = *(unsigned __int8 *)(v22 + 3);
        }
      }
      a2 = v24;
      LOBYTE(a2) = (unsigned __int8)v24 >> 6;
      if ( (((v24 & 0x10) != 0) & ((unsigned __int8)v24 >> 6)) == 0 )
        break;
      if ( !v59 )
      {
        v29 = v6 + 416;
        if ( (int)a4 >= 0 )
        {
          v48 = a4;
          a6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v58 + 16LL) + 200LL))(*(_QWORD *)(*v58 + 16LL));
          a4 = v48;
          a2 = *(_QWORD *)(*(_QWORD *)(a6 + 248) + 16LL);
          if ( *(_BYTE *)(a2 + v48) )
          {
            v30 = 1LL << v48;
            a2 = v48 >> 6;
            a4 = *(_QWORD *)(v58[48] + 8 * a2) & (1LL << v48);
            if ( !a4 )
            {
              a5 = v6 + 416;
              a2 = *(_QWORD *)(v57 + 56);
              a4 = *(_DWORD *)(*(_QWORD *)(v57 + 8) + 24LL * v48 + 16) >> 12;
              v31 = *(_DWORD *)(*(_QWORD *)(v57 + 8) + 24LL * v48 + 16) & 0xFFF;
              v32 = a2 + 2 * a4;
              do
              {
                if ( !v32 )
                  break;
                v32 += 2;
                v44 = v31;
                v49 = a5;
                sub_2F747D0(a5, a2, v30, a4, a5, a6, v31, -1, -1);
                a5 = v49;
                a2 = (unsigned int)*(__int16 *)(v32 - 2);
                v31 = a2 + v44;
              }
              while ( *(_WORD *)(v32 - 2) );
            }
          }
          goto LABEL_95;
        }
LABEL_94:
        sub_2F747D0(v29, a2, a3, a4, v29, a6, a4, -1, -1);
LABEL_95:
        v6 = v56;
      }
LABEL_43:
      a3 = v22 + 40;
      v26 = v23;
      if ( v22 + 40 == v23 )
      {
        while ( 1 )
        {
          v7 = *(_QWORD *)(v7 + 8);
          if ( v21 == v7 )
          {
            v22 = v23;
            v23 = v26;
            goto LABEL_49;
          }
          if ( (*(_BYTE *)(v7 + 44) & 4) == 0 )
            break;
          v23 = *(_QWORD *)(v7 + 32);
          v26 = v23 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
          if ( v23 != v26 )
            goto LABEL_53;
        }
        v22 = v23;
        v7 = v21;
        v23 = v26;
LABEL_49:
        if ( v23 == v22 )
          goto LABEL_50;
      }
      else
      {
        v23 = v22 + 40;
LABEL_53:
        v22 = v23;
        v23 = v26;
      }
    }
    v29 = v6 + 208;
    if ( (int)a4 >= 0 )
    {
      v53 = a4;
      a6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v58 + 16LL) + 200LL))(*(_QWORD *)(*v58 + 16LL));
      a4 = v53;
      a2 = *(_QWORD *)(*(_QWORD *)(a6 + 248) + 16LL);
      if ( *(_BYTE *)(a2 + v53) )
      {
        a2 = v53 >> 6;
        a4 = *(_QWORD *)(v58[48] + 8 * a2) & (1LL << v53);
        if ( !a4 )
        {
          a5 = v6 + 208;
          v40 = *(_QWORD *)(v57 + 8);
          a2 = *(_QWORD *)(v57 + 56);
          a4 = *(_DWORD *)(v40 + 24LL * v53 + 16) >> 12;
          v41 = *(_DWORD *)(v40 + 24LL * v53 + 16) & 0xFFF;
          v42 = a2 + 2 * a4;
          do
          {
            if ( !v42 )
              break;
            v42 += 2;
            v47 = v41;
            v54 = a5;
            sub_2F747D0(a5, a2, v40, a4, a5, a6, v41, -1, -1);
            a5 = v54;
            a2 = (unsigned int)*(__int16 *)(v42 - 2);
            v41 = a2 + v47;
          }
          while ( *(_WORD *)(v42 - 2) );
        }
      }
      goto LABEL_95;
    }
    goto LABEL_94;
  }
  if ( v9 )
  {
    do
      v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v7 + 44) & 4) != 0 );
  }
  v10 = *(_QWORD *)(a2 + 24) + 48LL;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v7 + 32);
    v12 = v11 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
    if ( v11 != v12 )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( v10 == v7 )
      break;
    if ( (*(_BYTE *)(v7 + 44) & 4) == 0 )
    {
      v7 = *(_QWORD *)(a2 + 24) + 48LL;
      break;
    }
  }
  v13 = 0;
  while ( v11 != v12 )
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)v11 )
      {
        a2 = *(unsigned int *)(v11 + 8);
        if ( (_DWORD)a2 )
        {
          a4 = *(unsigned __int8 *)(v11 + 4);
          v14 = *(_BYTE *)(v11 + 3);
          a3 = (*(_DWORD *)v11 >> 8) & 0xFFF;
          v15 = *(_BYTE *)(v11 + 4) & 1;
          if ( (v14 & 0x10) != 0 )
          {
            a4 = *(unsigned __int8 *)(v11 + 3);
            if ( v15 )
              a3 = 0;
            LOBYTE(a4) = (unsigned __int8)a4 >> 6;
            if ( (((v14 & 0x10) != 0) & (unsigned __int8)a4) != 0 )
            {
              if ( !v59 )
              {
                sub_2F74980((__int64)&v56, a2, a3, v6 + 416, 0, a6);
                v6 = v56;
                v13 = 0;
              }
            }
            else
            {
              sub_2F74980((__int64)&v56, a2, a3, v6 + 208, 0, a6);
              v6 = v56;
              v13 = 0;
            }
          }
          else if ( !v15 )
          {
            a4 &= 2u;
            if ( !(_DWORD)a4 )
            {
              sub_2F74980((__int64)&v56, a2, a3, v6, 0, a6);
              v6 = v56;
              v13 = 0;
            }
          }
        }
      }
      v16 = v11 + 40;
      v17 = v12;
      if ( v16 == v12 )
        break;
      v12 = v16;
LABEL_23:
      v11 = v12;
      v12 = v17;
    }
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( v10 == v7 )
      {
        v11 = v12;
        v12 = v17;
        goto LABEL_19;
      }
      if ( (*(_BYTE *)(v7 + 44) & 4) == 0 )
        break;
      v12 = *(_QWORD *)(v7 + 32);
      v17 = v12 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
      if ( v12 != v17 )
        goto LABEL_23;
    }
    v11 = v12;
    v7 = v10;
    v12 = v17;
LABEL_19:
    ;
  }
  v18 = *(_QWORD **)(v6 + 208);
  result = (__m128i *)(3LL * *(unsigned int *)(v6 + 216));
  if ( v18 != &v18[3 * *(unsigned int *)(v6 + 216)] )
  {
    while ( 1 )
    {
      result = sub_2F74650(v6 + 416, a2, a3, a4, v13, a6, *v18, v18[1], v18[2]);
      if ( v20 == v18 )
        break;
      v6 = v56;
    }
  }
  return result;
}
