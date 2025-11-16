// Function: sub_34B48B0
// Address: 0x34b48b0
//
__int64 __fastcall sub_34B48B0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // r15
  _BYTE *v5; // r14
  unsigned int v6; // r13d
  __int64 result; // rax
  _BYTE *v8; // r12
  _BYTE *v9; // rbx
  unsigned int v10; // esi
  _BYTE *v11; // r14
  __int64 v12; // rbx
  unsigned int v13; // r13d
  __int64 v14; // r12
  unsigned int v15; // r14d
  int v16; // eax
  char v17; // al
  char *v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // r12d
  __int64 v21; // r14
  char *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // r15
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  _BYTE *v29; // r12
  _BYTE *v30; // rbx
  _BYTE *v31; // r13
  unsigned int v32; // r14d
  __int64 v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  char *v36; // r8
  __int64 v37; // rdx
  char *v38; // r12
  __int64 v39; // rax
  unsigned __int16 v40; // bx
  __int16 *v41; // rax
  int v42; // esi
  __int64 v43; // rax
  __int16 v44; // di
  __int32 v45; // esi
  __int64 v46; // rax
  _BYTE *v47; // rbx
  int v48; // eax
  int v49; // eax
  __int64 v50; // rdi
  __int64 (*v51)(); // rax
  _BYTE *v52; // [rsp+0h] [rbp-C0h]
  __int64 v53; // [rsp+10h] [rbp-B0h]
  unsigned int v56; // [rsp+2Ch] [rbp-94h]
  __int64 v57; // [rsp+30h] [rbp-90h]
  unsigned int v58; // [rsp+40h] [rbp-80h]
  unsigned __int16 *v59; // [rsp+40h] [rbp-80h]
  unsigned int v61; // [rsp+5Ch] [rbp-64h] BYREF
  __m128i v62; // [rsp+60h] [rbp-60h] BYREF
  __int16 v63; // [rsp+70h] [rbp-50h]
  int v64; // [rsp+78h] [rbp-48h]
  __int64 v65; // [rsp+80h] [rbp-40h]
  __int16 v66; // [rsp+88h] [rbp-38h]

  v4 = a1;
  v5 = *(_BYTE **)(a2 + 32);
  v53 = *(_QWORD *)(a1 + 120);
  v6 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  result = 5LL * v6;
  v8 = &v5[40 * v6];
  if ( v5 != v8 )
  {
    while ( 1 )
    {
      v9 = v5;
      result = sub_2DADC00(v5);
      if ( (_BYTE)result )
        break;
      v5 += 40;
      if ( v8 == v5 )
        goto LABEL_14;
    }
    if ( v8 != v5 )
    {
      do
      {
        v10 = *((_DWORD *)v9 + 2);
        if ( v10 )
          sub_34B45F0(a1, v10, a3 + 1);
        v11 = v9 + 40;
        if ( v9 + 40 == v8 )
          break;
        while ( 1 )
        {
          v9 = v11;
          if ( sub_2DADC00(v11) )
            break;
          v11 += 40;
          if ( v8 == v11 )
            goto LABEL_13;
        }
      }
      while ( v8 != v11 );
LABEL_13:
      result = *(unsigned int *)(a2 + 40);
      v6 = result & 0xFFFFFF;
    }
  }
LABEL_14:
  if ( v6 )
  {
    v58 = v6;
    v12 = 0;
    result = *(_QWORD *)(a2 + 32);
    do
    {
      while ( 1 )
      {
        v13 = v12;
        v14 = result + 40 * v12;
        if ( !*(_BYTE *)v14 && (*(_BYTE *)(v14 + 3) & 0x10) != 0 )
        {
          v15 = *(_DWORD *)(v14 + 8);
          if ( v15 )
            break;
        }
        if ( v58 == (_DWORD)++v12 )
          goto LABEL_34;
      }
      v16 = *(_DWORD *)(a2 + 44);
      if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
        v17 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
      else
        v17 = sub_2E88A90(a2, 128, 1);
      if ( v17
        || ((v48 = *(_DWORD *)(a2 + 44), (v48 & 4) == 0) && (v48 & 8) != 0
          ? (LOBYTE(v49) = sub_2E88A90(a2, 0x100000000LL, 1))
          : (v49 = *(_DWORD *)(*(_QWORD *)(a2 + 16) + 28LL) & 1),
            (_BYTE)v49
         || (v50 = *(_QWORD *)(v4 + 24), v51 = *(__int64 (**)())(*(_QWORD *)v50 + 920LL), v51 != sub_2DB1B30)
         && ((unsigned __int8 (__fastcall *)(__int64, __int64))v51)(v50, a2)
         || (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1) )
      {
        sub_34B3410(*(_QWORD **)(v4 + 120), v15, 0);
      }
      v18 = sub_E922F0(*(_QWORD **)(v4 + 32), v15);
      if ( v18 != &v18[2 * v19 - 2] )
      {
        v57 = v14;
        v20 = v15;
        v21 = v12;
        v22 = v18;
        v56 = v13;
        v23 = v4;
        v24 = (__int64)&v18[2 * v19 - 2];
        do
        {
          v25 = *(_QWORD **)(v23 + 120);
          v26 = *(unsigned __int16 *)v22;
          if ( *(_DWORD *)(v25[13] + 4 * v26) != -1 && *(_DWORD *)(v25[16] + 4 * v26) == -1 )
            sub_34B3410(v25, v20, *(unsigned __int16 *)v22);
          v22 += 2;
        }
        while ( (char *)v24 != v22 );
        v12 = v21;
        v4 = v23;
        v15 = v20;
        v13 = v56;
        v14 = v57;
      }
      v27 = *(_QWORD *)(a2 + 16);
      if ( *(unsigned __int16 *)(v27 + 2) > v13 )
        v28 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v4 + 24) + 16LL))(
                *(_QWORD *)(v4 + 24),
                v27,
                v13,
                *(_QWORD *)(v4 + 32),
                *(_QWORD *)(v4 + 8));
      else
        v28 = 0;
      v62.m128i_i64[1] = v28;
      ++v12;
      v62.m128i_i64[0] = v14;
      v61 = v15;
      sub_34B43E0((_QWORD *)(v53 + 56), &v61, &v62);
      result = *(_QWORD *)(a2 + 32);
    }
    while ( v58 != (_DWORD)v12 );
LABEL_34:
    v29 = (_BYTE *)(result + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
    if ( (_BYTE *)result != v29 )
    {
      v30 = (_BYTE *)result;
      while ( 1 )
      {
        v31 = v30;
        result = sub_2DADC00(v30);
        if ( (_BYTE)result )
          break;
        v30 += 40;
        if ( v29 == v30 )
          return result;
      }
      if ( v29 != v30 )
      {
        result = a4 + 8;
        do
        {
          v32 = *((_DWORD *)v31 + 2);
          if ( v32 )
          {
            result = a2;
            if ( *(_WORD *)(a2 + 68) != 7 )
            {
              result = *(_QWORD *)(a4 + 16);
              if ( !result )
                goto LABEL_49;
              v33 = a4 + 8;
              do
              {
                while ( 1 )
                {
                  v34 = *(_QWORD *)(result + 16);
                  v35 = *(_QWORD *)(result + 24);
                  if ( v32 <= *(_DWORD *)(result + 32) )
                    break;
                  result = *(_QWORD *)(result + 24);
                  if ( !v35 )
                    goto LABEL_47;
                }
                v33 = result;
                result = *(_QWORD *)(result + 16);
              }
              while ( v34 );
LABEL_47:
              if ( a4 + 8 == v33 || v32 < *(_DWORD *)(v33 + 32) )
              {
LABEL_49:
                v36 = sub_E922F0(*(_QWORD **)(v4 + 32), v32);
                result = (__int64)&v36[2 * v37];
                v59 = (unsigned __int16 *)result;
                if ( v36 != (char *)result )
                {
                  v52 = v29;
                  v38 = v36;
                  do
                  {
                    v39 = *(_QWORD *)(v4 + 32);
                    v61 = *(unsigned __int16 *)v38;
                    v40 = v61;
                    v41 = (__int16 *)(*(_QWORD *)(v39 + 56)
                                    + 2LL * *(unsigned int *)(*(_QWORD *)(v39 + 8) + 24LL * v32 + 8));
                    v42 = *v41;
                    v43 = (__int64)(v41 + 1);
                    v64 = 0;
                    v65 = 0;
                    v44 = v42;
                    v45 = v32 + v42;
                    v62.m128i_i32[0] = v45;
                    if ( !v44 )
                      v43 = 0;
                    v63 = v45;
                    v62.m128i_i64[1] = v43;
                    v66 = 0;
                    if ( !sub_2E46590(v62.m128i_i32, (int *)&v61)
                      || (v46 = *(_QWORD *)(v4 + 120), *(_DWORD *)(*(_QWORD *)(v46 + 104) + 4LL * v40) == -1)
                      || (result = *(_QWORD *)(v46 + 128), *(_DWORD *)(result + 4LL * v40) != -1) )
                    {
                      result = *(_QWORD *)(v53 + 128);
                      *(_DWORD *)(result + 4LL * v40) = a3;
                    }
                    v38 += 2;
                  }
                  while ( v59 != (unsigned __int16 *)v38 );
                  v29 = v52;
                }
              }
            }
          }
          v47 = v31 + 40;
          if ( v31 + 40 == v29 )
            break;
          while ( 1 )
          {
            v31 = v47;
            result = sub_2DADC00(v47);
            if ( (_BYTE)result )
              break;
            v47 += 40;
            if ( v29 == v47 )
              return result;
          }
        }
        while ( v47 != v29 );
      }
    }
  }
  return result;
}
