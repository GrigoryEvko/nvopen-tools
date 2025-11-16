// Function: sub_2F41E70
// Address: 0x2f41e70
//
__int64 __fastcall sub_2F41E70(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 result; // rax
  unsigned int v9; // esi
  unsigned int *v10; // r13
  char v11; // al
  unsigned int v12; // edx
  unsigned int v13; // eax
  unsigned int v14; // edx
  unsigned int v15; // esi
  unsigned int *v16; // r13
  unsigned int *v17; // r15
  unsigned int *v18; // r14
  unsigned int v19; // edx
  int v20; // ecx
  int v21; // r8d
  int v22; // r9d
  unsigned int v23; // eax
  __int128 v24; // xmm0
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned int v29; // ecx
  unsigned int v30; // eax
  bool v31; // zf
  __int128 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+20h] [rbp-80h]
  unsigned int *v35; // [rsp+28h] [rbp-78h]
  __int64 v36; // [rsp+40h] [rbp-60h]
  __int128 v37; // [rsp+50h] [rbp-50h] BYREF
  __int64 v38; // [rsp+60h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  v35 = a2;
  v34 = a3;
  if ( (char *)a2 - (char *)a1 <= 64 )
    return result;
  if ( !a3 )
  {
    v18 = a2;
    goto LABEL_20;
  }
  v33 = a8;
  v32 = a7;
  while ( 2 )
  {
    v9 = a1[1];
    --v34;
    v10 = &a1[result >> 3];
    v37 = v32;
    v38 = v33;
    v11 = sub_2F41AD0(&v37, v9, *v10);
    v12 = *(v35 - 1);
    if ( !v11 )
    {
      if ( !sub_2F41AD0(&v37, a1[1], v12) )
      {
        v31 = sub_2F41AD0(&v37, *v10, *(v35 - 1)) == 0;
        v13 = *a1;
        if ( !v31 )
        {
          *a1 = *(v35 - 1);
          *(v35 - 1) = v13;
          v14 = *a1;
          v15 = a1[1];
          goto LABEL_8;
        }
        goto LABEL_7;
      }
LABEL_18:
      v15 = *a1;
      v14 = a1[1];
      a1[1] = *a1;
      *a1 = v14;
      goto LABEL_8;
    }
    if ( !sub_2F41AD0(&v37, *v10, v12) )
    {
      if ( sub_2F41AD0(&v37, a1[1], *(v35 - 1)) )
      {
        v30 = *a1;
        *a1 = *(v35 - 1);
        *(v35 - 1) = v30;
        v14 = *a1;
        v15 = a1[1];
        goto LABEL_8;
      }
      goto LABEL_18;
    }
    v13 = *a1;
LABEL_7:
    *a1 = *v10;
    *v10 = v13;
    v14 = *a1;
    v15 = a1[1];
LABEL_8:
    v16 = a1 + 1;
    v17 = v35;
    v37 = v32;
    v38 = v33;
    while ( 1 )
    {
      v18 = v16;
      if ( sub_2F41AD0(&v37, v15, v14) )
        goto LABEL_9;
      do
        v19 = *--v17;
      while ( sub_2F41AD0(&v37, *a1, v19) );
      if ( v16 >= v17 )
        break;
      v23 = *v16;
      *v16 = *v17;
      *v17 = v23;
LABEL_9:
      v14 = *a1;
      v15 = v16[1];
      ++v16;
    }
    sub_2F41E70((_DWORD)v16, (_DWORD)v35, v34, v20, v21, v22, a7, a8);
    result = (char *)v16 - (char *)a1;
    if ( (char *)v16 - (char *)a1 > 64 )
    {
      if ( v34 )
      {
        v35 = v16;
        continue;
      }
LABEL_20:
      v24 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
      v25 = result >> 2;
      v26 = ((result >> 2) - 2) >> 1;
      v38 = a8;
      v37 = v24;
      v36 = a8;
      sub_2F41CE0((__int64)a1, v26, result >> 2, a1[v26], a5, a6, v24, a8);
      do
      {
        --v26;
        sub_2F41CE0((__int64)a1, v26, v25, a1[v26], v27, v28, v37, v38);
      }
      while ( v26 );
      do
      {
        v29 = *--v18;
        *v18 = *a1;
        result = sub_2F41CE0((__int64)a1, 0, v18 - a1, v29, v27, v28, v24, v36);
      }
      while ( (char *)v18 - (char *)a1 > 4 );
    }
    return result;
  }
}
