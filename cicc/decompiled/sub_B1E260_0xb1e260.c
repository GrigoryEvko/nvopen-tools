// Function: sub_B1E260
// Address: 0xb1e260
//
__int64 __fastcall sub_B1E260(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // rsi
  __int64 v4; // rbx
  unsigned int v5; // r14d
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  _QWORD *v11; // r15
  __int64 result; // rax
  unsigned int v13; // r13d
  _QWORD *v14; // rbx
  __int64 v15; // r15
  unsigned int *v16; // r10
  unsigned int *v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // r12
  unsigned int v20; // ebx
  _QWORD *v21; // r13
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *i; // rax
  __int64 v28; // r14
  __int64 v29; // r15
  unsigned int v30; // r12d
  __int64 v31; // [rsp+8h] [rbp-1C8h]
  unsigned int v32; // [rsp+1Ch] [rbp-1B4h]
  __int64 v33; // [rsp+20h] [rbp-1B0h]
  unsigned int *v34; // [rsp+38h] [rbp-198h]
  __int64 v35; // [rsp+38h] [rbp-198h]
  _QWORD *v36; // [rsp+40h] [rbp-190h] BYREF
  __int64 v37; // [rsp+48h] [rbp-188h]
  _QWORD v38[8]; // [rsp+50h] [rbp-180h] BYREF
  _BYTE *v39; // [rsp+90h] [rbp-140h] BYREF
  __int64 v40; // [rsp+98h] [rbp-138h]
  _BYTE v41[304]; // [rsp+A0h] [rbp-130h] BYREF

  v1 = a1;
  v2 = *(unsigned int *)(a1 + 8);
  v36 = v38;
  v3 = 0x800000001LL;
  v32 = v2;
  v37 = 0x800000001LL;
  v38[0] = 0;
  if ( (unsigned int)v2 > 8 )
  {
    sub_C8D5F0(&v36, v38, v2, 8);
  }
  else if ( (unsigned int)v2 <= 1 )
  {
    v39 = v41;
    v40 = 0x2000000000LL;
    result = (unsigned int)(v2 - 1);
    if ( (unsigned int)result <= 1 )
      return result;
    v11 = v38;
    goto LABEL_8;
  }
  v4 = 8;
  v5 = 1;
  do
  {
    v3 = *(_QWORD *)(*(_QWORD *)a1 + v4);
    v6 = sub_B1E0B0(a1, v3);
    v7 = HIDWORD(v37);
    v8 = v6;
    *(_QWORD *)(v6 + 16) = *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(v6 + 4));
    v9 = (unsigned int)v37;
    v10 = (unsigned int)v37 + 1LL;
    if ( v10 > v7 )
    {
      v3 = (__int64)v38;
      sub_C8D5F0(&v36, v38, v10, 8);
      v9 = (unsigned int)v37;
    }
    ++v5;
    v4 += 8;
    v36[v9] = v8;
    LODWORD(v37) = v37 + 1;
  }
  while ( v5 < v32 );
  v1 = a1;
  v11 = v36;
  v39 = v41;
  v40 = 0x2000000000LL;
  result = v32 - 1;
  if ( (unsigned int)result > 1 )
  {
LABEL_8:
    v13 = v32;
    v31 = v1;
    v14 = v11;
    v33 = result;
    while ( 1 )
    {
      v15 = v14[v33];
      v16 = *(unsigned int **)(v15 + 24);
      *(_DWORD *)(v15 + 8) = *(_DWORD *)(v15 + 4);
      v17 = v16;
      result = (__int64)&v16[*(unsigned int *)(v15 + 32)];
      v34 = (unsigned int *)result;
      if ( (unsigned int *)result != v16 )
      {
        v18 = v14;
        v19 = v15;
        v20 = v13;
        v21 = v18;
        do
        {
          v22 = v21[*v17];
          v23 = (unsigned int)v40;
          if ( *(_DWORD *)(v22 + 4) < v20 )
          {
            v26 = *(unsigned int *)(v22 + 12);
          }
          else
          {
            do
            {
              if ( v23 + 1 > (unsigned __int64)HIDWORD(v40) )
              {
                sub_C8D5F0(&v39, v41, v23 + 1, 8);
                v23 = (unsigned int)v40;
              }
              *(_QWORD *)&v39[8 * v23] = v22;
              v23 = (unsigned int)(v40 + 1);
              LODWORD(v40) = v40 + 1;
              v22 = v21[*(unsigned int *)(v22 + 4)];
            }
            while ( *(_DWORD *)(v22 + 4) >= v20 );
            v24 = v21[*(unsigned int *)(v22 + 12)];
            do
            {
              while ( 1 )
              {
                v3 = (__int64)v39;
                v25 = v22;
                v22 = *(_QWORD *)&v39[8 * (unsigned int)v23 - 8];
                LODWORD(v40) = v23 - 1;
                *(_DWORD *)(v22 + 4) = *(_DWORD *)(v25 + 4);
                if ( *(_DWORD *)(v24 + 8) >= *(_DWORD *)(v21[*(unsigned int *)(v22 + 12)] + 8LL) )
                  break;
                *(_DWORD *)(v22 + 12) = *(_DWORD *)(v25 + 12);
                LODWORD(v23) = v40;
                if ( !(_DWORD)v40 )
                  goto LABEL_19;
              }
              v24 = v21[*(unsigned int *)(v22 + 12)];
              LODWORD(v23) = v40;
            }
            while ( (_DWORD)v40 );
LABEL_19:
            v26 = *(unsigned int *)(v22 + 12);
            v21 = v36;
          }
          result = *(unsigned int *)(v21[v26] + 8LL);
          if ( *(_DWORD *)(v19 + 8) > (unsigned int)result )
            *(_DWORD *)(v19 + 8) = result;
          ++v17;
        }
        while ( v34 != v17 );
        v13 = v20;
      }
      --v13;
      --v33;
      if ( v13 == 2 )
        break;
      v14 = v36;
    }
    if ( v32 > 2 )
    {
      v35 = 2;
      for ( i = v36; ; i = v36 )
      {
        v28 = i[v35];
        v29 = *(_QWORD *)(v28 + 16);
        v30 = *(_DWORD *)i[*(unsigned int *)(v28 + 8)];
        while ( 1 )
        {
          v3 = v29;
          result = sub_B1E0B0(v31, v29);
          if ( *(_DWORD *)result <= v30 )
            break;
          v29 = *(_QWORD *)(result + 16);
        }
        ++v35;
        ++v13;
        *(_QWORD *)(v28 + 16) = v29;
        if ( v32 <= v13 )
          break;
      }
    }
    if ( v39 != v41 )
      result = _libc_free(v39, v3);
    v11 = v36;
  }
  if ( v11 != v38 )
    return _libc_free(v11, v3);
  return result;
}
