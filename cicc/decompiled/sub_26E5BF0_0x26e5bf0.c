// Function: sub_26E5BF0
// Address: 0x26e5bf0
//
unsigned __int64 *__fastcall sub_26E5BF0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5)
{
  __int128 v6; // rax
  void *v7; // rbx
  size_t v8; // rdx
  size_t v9; // r14
  int v10; // eax
  unsigned int v11; // r8d
  __int64 *v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned __int64 *v15; // r14
  __int64 v16; // r10
  unsigned int v17; // r13d
  _DWORD *v18; // rax
  __int64 v19; // rax
  _DWORD *v20; // rax
  unsigned __int64 *v21; // rdx
  size_t v22; // r11
  __int64 v23; // r15
  unsigned __int64 *result; // rax
  __int64 v25; // rax
  int v26; // edx
  _DWORD *v27; // rax
  const void *v28; // rdi
  const void *v29; // rsi
  int v30; // eax
  _DWORD *v31; // rax
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rax
  unsigned int v35; // r8d
  __int64 *v36; // rcx
  __int64 v37; // rbx
  __int64 *v38; // rax
  __int64 *v39; // rax
  void *src; // [rsp+8h] [rbp-68h]
  __int64 *v41; // [rsp+10h] [rbp-60h]
  unsigned int v42; // [rsp+18h] [rbp-58h]
  unsigned int v43; // [rsp+18h] [rbp-58h]
  unsigned int v44; // [rsp+18h] [rbp-58h]
  __int64 v46; // [rsp+20h] [rbp-50h]
  int v48; // [rsp+34h] [rbp-3Ch] BYREF
  unsigned __int64 v49[7]; // [rsp+38h] [rbp-38h] BYREF

  *(_QWORD *)&v6 = sub_BD5D20(a2);
  v7 = (void *)sub_C16140(v6, (__int64)"selected", 8);
  v9 = v8;
  src = v7;
  v10 = sub_C92610();
  v11 = sub_C92740(a1 + 120, v7, v9, v10);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 120) + 8LL * v11);
  v13 = *v12;
  if ( *v12 )
  {
    if ( v13 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 136);
  }
  v41 = v12;
  v44 = v11;
  v34 = sub_C7D670(v9 + 65, 8);
  v35 = v44;
  v36 = v41;
  v37 = v34;
  if ( v9 )
  {
    memcpy((void *)(v34 + 64), src, v9);
    v35 = v44;
    v36 = v41;
  }
  *(_BYTE *)(v37 + v9 + 64) = 0;
  *(_OWORD *)(v37 + 40) = 0;
  *(_QWORD *)v37 = v9;
  *(_QWORD *)(v37 + 56) = 0;
  *(_QWORD *)(v37 + 8) = v37 + 56;
  *(_QWORD *)(v37 + 16) = 1;
  *(_DWORD *)(v37 + 40) = 1065353216;
  *(_QWORD *)(v37 + 48) = 0;
  *(_OWORD *)(v37 + 24) = 0;
  *v36 = v37;
  ++*(_DWORD *)(a1 + 132);
  v38 = (__int64 *)(*(_QWORD *)(a1 + 120) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 120), v35));
  v13 = *v38;
  if ( !*v38 || v13 == -8 )
  {
    v39 = v38 + 1;
    do
    {
      do
        v13 = *v39++;
      while ( v13 == -8 );
    }
    while ( !v13 );
  }
LABEL_3:
  v14 = *(_QWORD *)(a3 + 24);
  v15 = a4 + 1;
  v46 = a3 + 8;
  if ( v14 != v46 )
  {
    while ( 1 )
    {
      v16 = *(unsigned int *)(v14 + 36);
      v17 = *(_DWORD *)(v14 + 32);
      if ( a5 )
      {
        v42 = *(_DWORD *)(v14 + 36);
        v18 = sub_26E2B00(a5, *(_QWORD *)(v14 + 32) % a5[1], (_DWORD *)(v14 + 32), *(_QWORD *)(v14 + 32));
        v16 = v42;
        if ( v18 )
        {
          v19 = *(_QWORD *)v18;
          if ( v19 )
          {
            v17 = *(_DWORD *)(v19 + 16);
            v16 = *(unsigned int *)(v19 + 20);
          }
        }
      }
      v49[0] = __PAIR64__(v16, v17);
      v20 = (_DWORD *)a4[2];
      if ( v20 )
        break;
LABEL_19:
      v14 = sub_220EF30(v14);
      if ( v46 == v14 )
        goto LABEL_20;
    }
    v21 = a4 + 1;
    while ( 1 )
    {
      while ( v20[8] < v17 )
      {
        v20 = (_DWORD *)*((_QWORD *)v20 + 3);
LABEL_14:
        if ( !v20 )
        {
LABEL_15:
          if ( v15 != v21
            && *((_DWORD *)v21 + 8) <= v17
            && (*((_DWORD *)v21 + 8) != v17 || *((_DWORD *)v21 + 9) <= (unsigned int)v16) )
          {
            v22 = *(_QWORD *)(v14 + 48);
            if ( v22 == v21[6] )
            {
              v28 = *(const void **)(v14 + 40);
              v29 = (const void *)v21[5];
              if ( v28 == v29 || (v43 = v16, v28) && v29 && (v30 = memcmp(v28, v29, v22), v16 = v43, !v30) )
              {
                v31 = sub_26E3C30(
                        (_QWORD *)(v13 + 8),
                        ((v16 << 32) | (unsigned __int64)v17) % *(_QWORD *)(v13 + 16),
                        v49,
                        (v16 << 32) | v17);
                if ( v31 && (v32 = *(_QWORD *)v31) != 0 )
                {
                  if ( a5 )
                  {
                    v33 = *(_DWORD *)(v32 + 16);
                    if ( v33 == 1 )
                    {
                      *(_DWORD *)(v32 + 16) = 3;
                    }
                    else if ( v33 == 2 )
                    {
                      *(_DWORD *)(v32 + 16) = 5;
                    }
                  }
                }
                else
                {
                  v48 = 1;
                  sub_26E3CA0((unsigned __int64 *)(v13 + 8), v49, &v48);
                }
              }
            }
          }
          goto LABEL_19;
        }
      }
      if ( v20[8] == v17 && v20[9] < (unsigned int)v16 )
      {
        v20 = (_DWORD *)*((_QWORD *)v20 + 3);
        goto LABEL_14;
      }
      v21 = (unsigned __int64 *)v20;
      v20 = (_DWORD *)*((_QWORD *)v20 + 2);
      if ( !v20 )
        goto LABEL_15;
    }
  }
LABEL_20:
  v23 = a4[3];
  result = v49;
  if ( v15 != (unsigned __int64 *)v23 )
  {
    do
    {
      while ( 1 )
      {
        v27 = sub_26E3C30(
                (_QWORD *)(v13 + 8),
                *(_QWORD *)(v23 + 32) % *(_QWORD *)(v13 + 16),
                (_DWORD *)(v23 + 32),
                *(_QWORD *)(v23 + 32));
        if ( v27 )
        {
          v25 = *(_QWORD *)v27;
          if ( v25 )
            break;
        }
        LODWORD(v49[0]) = 2;
        sub_26E3CA0((unsigned __int64 *)(v13 + 8), (unsigned __int64 *)(v23 + 32), v49);
        result = (unsigned __int64 *)sub_220EF30(v23);
        v23 = (__int64)result;
        if ( v15 == result )
          return result;
      }
      if ( a5 )
      {
        v26 = *(_DWORD *)(v25 + 16);
        if ( v26 == 2 )
        {
          *(_DWORD *)(v25 + 16) = 4;
        }
        else if ( v26 == 1 )
        {
          *(_DWORD *)(v25 + 16) = 6;
        }
      }
      result = (unsigned __int64 *)sub_220EF30(v23);
      v23 = (__int64)result;
    }
    while ( v15 != result );
  }
  return result;
}
