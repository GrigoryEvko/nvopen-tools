// Function: sub_E655A0
// Address: 0xe655a0
//
__int64 __fastcall sub_E655A0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r9
  unsigned __int64 i; // rcx
  __int64 v5; // r8
  __int64 v6; // r12
  size_t v7; // rdi
  size_t v8; // rdx
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  _BYTE *v11; // r9
  size_t v12; // rdx
  size_t v13; // rdi
  size_t v14; // rdx
  const void *v15; // r8
  char *v16; // rsi
  _BYTE *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  size_t v20; // rdx
  __int64 v21; // [rsp-1B0h] [rbp-1B0h]
  _BYTE *v22; // [rsp-198h] [rbp-198h]
  __int64 v23; // [rsp-190h] [rbp-190h]
  size_t v24; // [rsp-188h] [rbp-188h]
  size_t v25; // [rsp-188h] [rbp-188h]
  const void *v26; // [rsp-188h] [rbp-188h]
  __int64 v27; // [rsp-180h] [rbp-180h]
  size_t v28; // [rsp-180h] [rbp-180h]
  size_t v29; // [rsp-180h] [rbp-180h]
  _QWORD *v30; // [rsp-178h] [rbp-178h] BYREF
  char *v31; // [rsp-170h] [rbp-170h]
  _QWORD v32[2]; // [rsp-168h] [rbp-168h] BYREF
  char *v33; // [rsp-158h] [rbp-158h] BYREF
  size_t v34; // [rsp-150h] [rbp-150h]
  __int64 v35; // [rsp-148h] [rbp-148h]
  _BYTE v36[320]; // [rsp-140h] [rbp-140h] BYREF

  result = *(unsigned int *)(a1 + 1688);
  if ( (_DWORD)result )
  {
    sub_E65530(a1, (char **)(a1 + 1528));
    v34 = 0;
    v33 = v36;
    result = *(_QWORD *)(a1 + 1752);
    v35 = 256;
    v23 = result;
    v21 = a1 + 1736;
    if ( result != a1 + 1736 )
    {
      for ( i = 256; ; i = v35 )
      {
        v5 = *(_QWORD *)(v23 + 48);
        v27 = v5 + 32LL * *(unsigned int *)(v23 + 56);
        if ( v27 != v5 )
        {
          v6 = *(_QWORD *)(v23 + 48);
          while ( 1 )
          {
            v8 = *(_QWORD *)(v6 + 8);
            v11 = *(_BYTE **)v6;
            v7 = 0;
            v34 = 0;
            if ( v8 > i )
            {
              v22 = v11;
              v25 = v8;
              sub_C8D290((__int64)&v33, v36, v8, 1u, v5, (__int64)v11);
              v7 = v34;
              v11 = v22;
              v8 = v25;
            }
            if ( v8 )
            {
              v24 = v8;
              memcpy(&v33[v7], v11, v8);
              v7 = v34;
              v8 = v24;
            }
            v34 = v8 + v7;
            sub_E65530(a1, &v33);
            v30 = v32;
            sub_E62BB0((__int64 *)&v30, v33, (__int64)&v33[v34]);
            v9 = *(_BYTE **)v6;
            if ( v30 == v32 )
            {
              v12 = (size_t)v31;
              if ( v31 )
              {
                if ( v31 == (char *)1 )
                  *v9 = v32[0];
                else
                  memcpy(v9, v32, (size_t)v31);
                v12 = (size_t)v31;
                v9 = *(_BYTE **)v6;
              }
              *(_QWORD *)(v6 + 8) = v12;
              v9[v12] = 0;
              v9 = v30;
              goto LABEL_12;
            }
            if ( v9 == (_BYTE *)(v6 + 16) )
              break;
            *(_QWORD *)v6 = v30;
            v10 = *(_QWORD *)(v6 + 16);
            *(_QWORD *)(v6 + 8) = v31;
            *(_QWORD *)(v6 + 16) = v32[0];
            if ( !v9 )
              goto LABEL_23;
            v30 = v9;
            v32[0] = v10;
LABEL_12:
            v31 = 0;
            *v9 = 0;
            if ( v30 != v32 )
              j_j___libc_free_0(v30, v32[0] + 1LL);
            i = v35;
            v6 += 32;
            if ( v27 == v6 )
              goto LABEL_24;
          }
          *(_QWORD *)v6 = v30;
          *(_QWORD *)(v6 + 8) = v31;
          *(_QWORD *)(v6 + 16) = v32[0];
LABEL_23:
          v9 = v32;
          v30 = v32;
          goto LABEL_12;
        }
LABEL_24:
        v13 = 0;
        v34 = 0;
        v14 = *(_QWORD *)(v23 + 480);
        v15 = *(const void **)(v23 + 472);
        if ( v14 > i )
        {
          v26 = *(const void **)(v23 + 472);
          v29 = *(_QWORD *)(v23 + 480);
          sub_C8D290((__int64)&v33, v36, v14, 1u, (__int64)v15, v3);
          v13 = v34;
          v15 = v26;
          v14 = v29;
        }
        if ( v14 )
        {
          v28 = v14;
          memcpy(&v33[v13], v15, v14);
          v13 = v34;
          v14 = v28;
        }
        v34 = v13 + v14;
        sub_E65530(a1, &v33);
        v16 = v33;
        v30 = v32;
        sub_E62BB0((__int64 *)&v30, v33, (__int64)&v33[v34]);
        v17 = *(_BYTE **)(v23 + 472);
        if ( v30 == v32 )
        {
          v20 = (size_t)v31;
          if ( v31 )
          {
            if ( v31 == (char *)1 )
            {
              *v17 = v32[0];
            }
            else
            {
              v16 = (char *)v32;
              memcpy(v17, v32, (size_t)v31);
            }
            v20 = (size_t)v31;
            v17 = *(_BYTE **)(v23 + 472);
          }
          *(_QWORD *)(v23 + 480) = v20;
          v17[v20] = 0;
          v17 = v30;
        }
        else
        {
          v18 = v32[0];
          v16 = v31;
          if ( v17 == (_BYTE *)(v23 + 488) )
          {
            *(_QWORD *)(v23 + 472) = v30;
            *(_QWORD *)(v23 + 480) = v16;
            *(_QWORD *)(v23 + 488) = v18;
          }
          else
          {
            v19 = *(_QWORD *)(v23 + 488);
            *(_QWORD *)(v23 + 472) = v30;
            *(_QWORD *)(v23 + 480) = v16;
            *(_QWORD *)(v23 + 488) = v18;
            if ( v17 )
            {
              v30 = v17;
              v32[0] = v19;
              goto LABEL_32;
            }
          }
          v17 = v32;
          v30 = v32;
        }
LABEL_32:
        v31 = 0;
        *v17 = 0;
        if ( v30 != v32 )
        {
          v16 = (char *)(v32[0] + 1LL);
          j_j___libc_free_0(v30, v32[0] + 1LL);
        }
        result = sub_220EEE0(v23);
        v23 = result;
        if ( v21 == result )
        {
          if ( v33 != v36 )
            return _libc_free(v33, v16);
          return result;
        }
      }
    }
  }
  return result;
}
