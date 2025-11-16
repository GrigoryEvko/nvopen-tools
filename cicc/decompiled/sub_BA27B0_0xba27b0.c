// Function: sub_BA27B0
// Address: 0xba27b0
//
_BYTE *__fastcall sub_BA27B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // r12d
  int v14; // r12d
  __int64 *v15; // r13
  __int64 v16; // r15
  _BYTE *v17; // rax
  int v18; // r13d
  unsigned int v20; // esi
  int v21; // eax
  _QWORD *v22; // rdx
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int8 v28; // al
  __int64 *v29; // rcx
  __int64 v30; // [rsp+0h] [rbp-A0h]
  int v31; // [rsp+10h] [rbp-90h]
  unsigned int i; // [rsp+14h] [rbp-8Ch]
  _BYTE *v33[2]; // [rsp+18h] [rbp-88h] BYREF
  _QWORD *v34; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v35; // [rsp+30h] [rbp-70h] BYREF
  __int64 v36; // [rsp+38h] [rbp-68h] BYREF
  __int64 v37; // [rsp+40h] [rbp-60h] BYREF
  __int64 v38; // [rsp+48h] [rbp-58h] BYREF
  __int64 v39; // [rsp+50h] [rbp-50h] BYREF
  __int64 v40; // [rsp+58h] [rbp-48h]
  int v41; // [rsp+60h] [rbp-40h]
  char v42; // [rsp+64h] [rbp-3Ch]

  v2 = a1;
  v3 = a1 - 16;
  v33[0] = (_BYTE *)a1;
  if ( *(_BYTE *)a1 != 16 )
    v2 = *(_QWORD *)sub_A17150((_BYTE *)(a1 - 16));
  v35 = (_QWORD *)v2;
  v5 = *(_BYTE *)(a1 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_QWORD *)(a1 - 32);
  else
    v6 = v3 - 8LL * ((v5 >> 2) & 0xF);
  v36 = *(_QWORD *)(v6 + 8);
  v7 = *(_BYTE *)(a1 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(a1 - 32);
  else
    v8 = v3 - 8LL * ((v7 >> 2) & 0xF);
  v37 = *(_QWORD *)(v8 + 16);
  v9 = *(_BYTE *)(a1 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(_QWORD *)(a1 - 32);
  else
    v10 = v3 - 8LL * ((v9 >> 2) & 0xF);
  v38 = *(_QWORD *)(v10 + 24);
  v39 = sub_AF5140(a1, 4u);
  v11 = sub_AF5140(a1, 5u);
  v12 = *(_QWORD *)(a2 + 8);
  v40 = v11;
  v41 = *(_DWORD *)(a1 + 4);
  v13 = *(_DWORD *)(a2 + 24);
  v42 = *(_BYTE *)(a1 + 1) >> 7;
  if ( v13 )
  {
    v14 = v13 - 1;
    v31 = 1;
    for ( i = v14 & sub_AFBE30(&v36, &v37, &v38, &v39); ; i = v18 )
    {
      v15 = (__int64 *)(v12 + 8LL * i);
      v16 = *v15;
      if ( *v15 == -4096 )
        break;
      if ( v16 != -8192 )
      {
        v17 = sub_A17150((_BYTE *)(v16 - 16));
        if ( v36 == *((_QWORD *)v17 + 1) )
        {
          v30 = v16;
          v24 = sub_AF5140(v16, 2u);
          if ( v37 == v24 )
          {
            v25 = sub_AF5140(v16, 3u);
            if ( v38 == v25 )
            {
              v26 = sub_AF5140(v16, 4u);
              if ( v39 == v26 )
              {
                v27 = sub_AF5140(v16, 5u);
                if ( v40 == v27 )
                {
                  if ( *(_BYTE *)v16 != 16 )
                  {
                    v28 = *(_BYTE *)(v16 - 16);
                    if ( (v28 & 2) != 0 )
                      v29 = *(__int64 **)(v16 - 32);
                    else
                      v29 = (__int64 *)(v16 - 16 - 8LL * ((v28 >> 2) & 0xF));
                    v30 = *v29;
                  }
                  if ( v35 == (_QWORD *)v30
                    && v41 == *(_DWORD *)(v16 + 4)
                    && v42 == (unsigned __int8)BYTE1(*(_QWORD *)v16) >> 7 )
                  {
                    if ( v15 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
                      break;
                    return (_BYTE *)v16;
                  }
                }
              }
            }
          }
        }
      }
      v18 = v14 & (v31 + i);
      ++v31;
    }
  }
  if ( !(unsigned __int8)sub_AFE760(a2, v33, &v34) )
  {
    v20 = *(_DWORD *)(a2 + 24);
    v21 = *(_DWORD *)(a2 + 16);
    v22 = v34;
    ++*(_QWORD *)a2;
    v23 = v21 + 1;
    v35 = v22;
    if ( 4 * v23 >= 3 * v20 )
    {
      v20 *= 2;
    }
    else if ( v20 - *(_DWORD *)(a2 + 20) - v23 > v20 >> 3 )
    {
LABEL_22:
      *(_DWORD *)(a2 + 16) = v23;
      if ( *v22 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v22 = v33[0];
      return v33[0];
    }
    sub_B09F80(a2, v20);
    sub_AFE760(a2, v33, &v35);
    v22 = v35;
    v23 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_22;
  }
  return v33[0];
}
