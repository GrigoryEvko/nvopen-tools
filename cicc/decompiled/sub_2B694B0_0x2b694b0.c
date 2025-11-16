// Function: sub_2B694B0
// Address: 0x2b694b0
//
__int64 __fastcall sub_2B694B0(__int64 a1, unsigned int a2)
{
  __m128i *v2; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rsi
  __int64 v11; // r8
  int v13; // esi
  char **v14; // r15
  __int64 *v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 *v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // rdi
  int v25; // r10d
  char v26; // al
  __int64 v27; // rdi
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  char v30; // al
  __int64 v31; // rdi
  unsigned __int64 v32; // rax
  __int64 *v33; // [rsp+8h] [rbp-88h]
  __m128i v34; // [rsp+10h] [rbp-80h] BYREF
  __int64 v35; // [rsp+20h] [rbp-70h]
  __int64 v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+30h] [rbp-60h]
  __int64 v38; // [rsp+38h] [rbp-58h]
  __int64 v39; // [rsp+40h] [rbp-50h]
  __int64 v40; // [rsp+48h] [rbp-48h]
  __int16 v41; // [rsp+50h] [rbp-40h]

  v4 = sub_2B68AE0(*(_QWORD *)(a1 + 8), **(_QWORD **)a1, a2);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = v4;
  v7 = *(unsigned int *)(v5 + 3544);
  v8 = *(_QWORD *)(v5 + 3528);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v10 = (__int64 *)(v8 + 24LL * v9);
    v11 = *v10;
    if ( v6 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 24 * v7) )
      {
        LODWORD(v2) = *((unsigned __int8 *)v10 + 16);
        return (unsigned int)v2;
      }
    }
    else
    {
      v13 = 1;
      while ( v11 != -4096 )
      {
        v25 = v13 + 1;
        v9 = (v7 - 1) & (v13 + v9);
        v10 = (__int64 *)(v8 + 24LL * v9);
        v11 = *v10;
        if ( v6 == *v10 )
          goto LABEL_3;
        v13 = v25;
      }
    }
  }
  v14 = *(char ***)(a1 + 16);
  v15 = *(__int64 **)v6;
  v16 = 8LL * *(unsigned int *)(v6 + 8);
  v33 = (__int64 *)(*(_QWORD *)v6 + v16);
  v17 = v16 >> 3;
  v18 = v16 >> 5;
  if ( v18 )
  {
    v2 = &v34;
    v19 = &v15[4 * v18];
    do
    {
      if ( **v14 != 13 )
      {
        v24 = *v15;
        v34 = (__m128i)*(unsigned __int64 *)(v5 + 3344);
        v35 = 0;
        v36 = 0;
        v37 = 0;
        v38 = 0;
        v39 = 0;
        v40 = 0;
        v41 = 257;
        if ( !(unsigned __int8)sub_9AC470(v24, &v34, 0) )
          goto LABEL_19;
        v20 = v15[1];
        if ( **v14 != 13 )
        {
          v34 = (__m128i)*(unsigned __int64 *)(v5 + 3344);
          v35 = 0;
          v36 = 0;
          v37 = 0;
          v38 = 0;
          v39 = 0;
          v40 = 0;
          v41 = 257;
          if ( !(unsigned __int8)sub_9AC470(v20, &v34, 0) )
          {
            ++v15;
            goto LABEL_19;
          }
          v21 = v15[2];
          if ( **v14 != 13 )
          {
            v34 = (__m128i)*(unsigned __int64 *)(v5 + 3344);
            v35 = 0;
            v36 = 0;
            v37 = 0;
            v38 = 0;
            v39 = 0;
            v40 = 0;
            v41 = 257;
            if ( !(unsigned __int8)sub_9AC470(v21, &v34, 0) )
            {
              LOBYTE(v2) = v33 != v15 + 2;
              return (unsigned int)v2;
            }
            v22 = v15[3];
            if ( **v14 != 13 )
            {
              v23 = *(_QWORD *)(v5 + 3344);
              v41 = 257;
              v34 = (__m128i)v23;
              v35 = 0;
              v36 = 0;
              v37 = 0;
              v38 = 0;
              v39 = 0;
              v40 = 0;
              if ( !(unsigned __int8)sub_9AC470(v22, &v34, 0) )
              {
                LOBYTE(v2) = v33 != v15 + 3;
                return (unsigned int)v2;
              }
            }
          }
        }
      }
      v15 += 4;
    }
    while ( v15 != v19 );
    v17 = v33 - v15;
  }
  if ( v17 == 2 )
  {
    v30 = **v14;
    goto LABEL_38;
  }
  if ( v17 == 3 )
  {
    v26 = **v14;
    if ( v26 == 13 )
    {
      ++v15;
LABEL_41:
      ++v15;
      goto LABEL_31;
    }
    v28 = *v15;
    v29 = *(_QWORD *)(v5 + 3344);
    v35 = 0;
    v34 = (__m128i)v29;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 257;
    if ( !(unsigned __int8)sub_9AC470(v28, &v34, 0) )
    {
      LOBYTE(v2) = v15 != v33;
      return (unsigned int)v2;
    }
    ++v15;
    v30 = **v14;
LABEL_38:
    if ( v30 == 13 )
      goto LABEL_29;
    v31 = *v15;
    v32 = *(_QWORD *)(v5 + 3344);
    v41 = 257;
    v34 = (__m128i)v32;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    if ( !(unsigned __int8)sub_9AC470(v31, &v34, 0) )
    {
LABEL_19:
      LOBYTE(v2) = v33 != v15;
      return (unsigned int)v2;
    }
    v26 = **v14;
    goto LABEL_41;
  }
  if ( v17 != 1 )
  {
LABEL_29:
    LODWORD(v2) = 0;
    return (unsigned int)v2;
  }
  v26 = **v14;
LABEL_31:
  LODWORD(v2) = 0;
  if ( v26 != 13 )
  {
    v27 = *v15;
    v34 = (__m128i)*(unsigned __int64 *)(v5 + 3344);
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 257;
    if ( !(unsigned __int8)sub_9AC470(v27, &v34, 0) )
      goto LABEL_19;
  }
  return (unsigned int)v2;
}
