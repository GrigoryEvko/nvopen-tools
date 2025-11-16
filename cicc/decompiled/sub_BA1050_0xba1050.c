// Function: sub_BA1050
// Address: 0xba1050
//
__int64 __fastcall sub_BA1050(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r13
  __int64 v4; // rax
  _BYTE *v5; // rax
  int v6; // r13d
  __int64 v7; // r14
  __int8 *v8; // rax
  __int8 *v9; // rbx
  int v10; // ebx
  int v11; // ecx
  unsigned int i; // ebx
  __int64 *v13; // r13
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdi
  __int16 v17; // ax
  __int64 result; // rax
  unsigned int v19; // esi
  int v20; // eax
  __int64 *v21; // rdx
  int v22; // eax
  unsigned int v23; // ebx
  __int64 v24; // rdi
  __int64 v25; // rax
  _BYTE *v26; // rax
  _BYTE *v27; // [rsp+0h] [rbp-170h]
  __int64 v28; // [rsp+8h] [rbp-168h]
  int v29; // [rsp+8h] [rbp-168h]
  __int64 v30; // [rsp+8h] [rbp-168h]
  int v31; // [rsp+10h] [rbp-160h]
  __int64 v32; // [rsp+10h] [rbp-160h]
  int v33; // [rsp+10h] [rbp-160h]
  __int64 v34; // [rsp+18h] [rbp-158h]
  int v35; // [rsp+18h] [rbp-158h]
  __int64 v36; // [rsp+18h] [rbp-158h]
  __int64 v37; // [rsp+18h] [rbp-158h]
  int v38; // [rsp+20h] [rbp-150h]
  __int64 v39[2]; // [rsp+28h] [rbp-148h] BYREF
  __int64 v40; // [rsp+38h] [rbp-138h] BYREF
  __int64 v41; // [rsp+40h] [rbp-130h] BYREF
  __int64 v42; // [rsp+48h] [rbp-128h]
  __int64 *v43; // [rsp+50h] [rbp-120h] BYREF
  __int64 v44; // [rsp+58h] [rbp-118h] BYREF
  __int64 v45; // [rsp+60h] [rbp-110h] BYREF
  int v46; // [rsp+68h] [rbp-108h] BYREF
  _BYTE *v47; // [rsp+70h] [rbp-100h] BYREF
  __int64 v48[3]; // [rsp+78h] [rbp-F8h] BYREF
  int v49; // [rsp+90h] [rbp-E0h]
  __int64 v50; // [rsp+94h] [rbp-DCh]
  __int64 v51; // [rsp+9Ch] [rbp-D4h]
  int v52; // [rsp+A4h] [rbp-CCh] BYREF
  __int64 v53; // [rsp+A8h] [rbp-C8h]
  __int64 v54; // [rsp+B0h] [rbp-C0h]
  __m128i dest[4]; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int64 v56[7]; // [rsp+100h] [rbp-70h] BYREF
  __int64 (__fastcall *v57)(); // [rsp+138h] [rbp-38h]

  v3 = (_BYTE *)(a1 - 16);
  v39[0] = a1;
  LODWORD(v43) = (unsigned __int16)sub_AF18C0(a1);
  v44 = sub_AF5140(a1, 2u);
  v4 = a1;
  if ( *(_BYTE *)a1 != 16 )
    v4 = *(_QWORD *)sub_A17150(v3);
  v45 = v4;
  v46 = *(_DWORD *)(a1 + 16);
  v47 = (_BYTE *)*((_QWORD *)sub_A17150(v3) + 1);
  v48[0] = *((_QWORD *)sub_A17150(v3) + 3);
  v48[1] = *(_QWORD *)(a1 + 24);
  v48[2] = *(_QWORD *)(a1 + 32);
  v49 = sub_AF18D0(a1);
  v42 = *(_QWORD *)(a1 + 44);
  dest[0].m128i_i64[0] = v42;
  v50 = v42;
  v51 = sub_AF2E40(a1);
  v52 = *(_DWORD *)(a1 + 20);
  v53 = *((_QWORD *)sub_A17150(v3) + 4);
  v5 = sub_A17150(v3);
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 8);
  v54 = *((_QWORD *)v5 + 5);
  if ( v6 )
  {
    if ( (_DWORD)v43 == 13 && v44 && v47 && *v47 == 14 && sub_AF5140((__int64)v47, 7u) )
    {
      memset(dest, 0, sizeof(dest));
      memset(v56, 0, sizeof(v56));
      v57 = sub_C64CA0;
      v40 = 0;
      v8 = sub_AF8740(dest, &v40, dest[0].m128i_i8, (unsigned __int64)v56, v44);
      v41 = v40;
      v9 = sub_AF70F0(dest, &v41, v8, (unsigned __int64)v56, (__int64)v47);
      if ( v41 )
      {
        v37 = v41;
        sub_B8EB50(dest[0].m128i_i8, v9, (char *)v56);
        sub_AC2A10(v56, dest);
        v10 = sub_AF1490(v56, v9 - (__int8 *)dest + v37);
      }
      else
      {
        v10 = sub_AC25F0(dest, v9 - (__int8 *)dest, (__int64)v57);
      }
    }
    else
    {
      v10 = sub_AF95C0((int *)&v43, &v44, &v45, &v46, (__int64 *)&v47, v48, &v52);
    }
    v38 = 1;
    v11 = v6 - 1;
    for ( i = (v6 - 1) & v10; ; i = v11 & v23 )
    {
      v13 = (__int64 *)(v7 + 8LL * i);
      v14 = *v13;
      if ( *v13 == -4096 )
        break;
      if ( v14 != -8192 )
      {
        if ( v44 != 0 && (_DWORD)v43 == 13 )
        {
          if ( v47 )
          {
            v34 = v44;
            if ( *v47 == 14 )
            {
              v28 = *v13;
              v31 = v11;
              v27 = v47;
              v15 = sub_AF5140((__int64)v47, 7u);
              v11 = v31;
              v14 = v28;
              if ( v15 )
              {
                v16 = v28;
                v29 = v31;
                v32 = v14;
                v17 = sub_AF18C0(v16);
                v14 = v32;
                v11 = v29;
                if ( v17 == 13 )
                {
                  v24 = v32;
                  v30 = v34;
                  v33 = v11;
                  v36 = v14;
                  v25 = sub_AF5140(v24, 2u);
                  v14 = v36;
                  v11 = v33;
                  if ( v30 == v25 )
                  {
                    v26 = sub_A17150((_BYTE *)(v36 - 16));
                    v14 = v36;
                    v11 = v33;
                    if ( v27 == *((_BYTE **)v26 + 1) )
                      goto LABEL_22;
                  }
                }
              }
            }
          }
        }
        v35 = v11;
        if ( sub_AF5170((int *)&v43, v14) )
        {
LABEL_22:
          if ( v13 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
            break;
          result = *v13;
          if ( !*v13 )
            break;
          return result;
        }
        v11 = v35;
        v14 = *v13;
      }
      if ( v14 == -4096 )
        break;
      v23 = v38 + i;
      ++v38;
    }
  }
  if ( !(unsigned __int8)sub_AFD260(a2, v39, &v43) )
  {
    v19 = *(_DWORD *)(a2 + 24);
    v20 = *(_DWORD *)(a2 + 16);
    v21 = v43;
    ++*(_QWORD *)a2;
    v22 = v20 + 1;
    dest[0].m128i_i64[0] = (__int64)v21;
    if ( 4 * v22 >= 3 * v19 )
    {
      v19 *= 2;
    }
    else if ( v19 - *(_DWORD *)(a2 + 20) - v22 > v19 >> 3 )
    {
LABEL_30:
      *(_DWORD *)(a2 + 16) = v22;
      if ( *v21 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v21 = v39[0];
      return v39[0];
    }
    sub_B05890(a2, v19);
    sub_AFD260(a2, v39, (__int64 **)dest);
    v21 = (__int64 *)dest[0].m128i_i64[0];
    v22 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_30;
  }
  return v39[0];
}
