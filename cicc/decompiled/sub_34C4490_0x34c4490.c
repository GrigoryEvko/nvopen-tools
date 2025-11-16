// Function: sub_34C4490
// Address: 0x34c4490
//
__int64 __fastcall sub_34C4490(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rbx
  void *v6; // rdi
  __int64 v7; // rax
  int v8; // r12d
  __int64 *v9; // rbx
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 *v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // rsi
  __int64 *v15; // r11
  __int64 v16; // r9
  __int64 v17; // rsi
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 *v21; // r11
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned __int64 v31; // r13
  _QWORD *v32; // r12
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  __int64 v35; // rsi
  bool v37; // al
  bool v38; // al
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned int v43; // [rsp+18h] [rbp-58h]
  unsigned __int8 v44; // [rsp+20h] [rbp-50h]
  __int64 v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v47[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v5 = a1[1];
  v6 = (void *)*a1;
  v46 = a3;
  v7 = v5 - (_QWORD)v6;
  if ( v5 - (__int64)v6 > 24 )
  {
    qsort(v6, 0xAAAAAAAAAAAAAAABLL * (v7 >> 3), 0x18u, (__compar_fn_t)sub_34BE7B0);
    v5 = a1[1];
    v7 = v5 - *a1;
  }
  v44 = 0;
  if ( (unsigned __int64)v7 <= 0x18 )
    return v44;
  do
  {
    v8 = *(_DWORD *)(v5 - 24);
    v9 = (__int64 *)(v5 - 8);
    v10 = sub_34C2FF0(a1, v8, a4, a2, v46);
    v14 = (__int64 *)a1[14];
    v15 = (__int64 *)a1[13];
    v16 = v10;
    if ( v15 == v14 )
      goto LABEL_40;
    v17 = (char *)v14 - (char *)v15;
    v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 8) + 32LL) + 328LL);
    v47[0] = v17 >> 4;
    if ( v17 == 32 )
    {
      v43 = v10;
      v37 = sub_2E322F0(*(_QWORD *)(*v15 + 8), *(_QWORD *)(v15[2] + 8));
      v15 = (__int64 *)a1[13];
      v16 = v43;
      if ( v37 )
      {
        v40 = *(_QWORD *)(v15[2] + 8);
        v17 = a1[14] - (_QWORD)v15;
        v12 = *(__int64 **)(v40 + 56);
        if ( (__int64 *)v15[3] == v12 && !*(_BYTE *)(v40 + 216) )
        {
          v47[0] = 1;
          v19 = v17 >> 4;
          v20 = 1;
          goto LABEL_17;
        }
      }
      else
      {
        v17 = a1[14] - (_QWORD)v15;
      }
      if ( v17 == 32 )
      {
        v38 = sub_2E322F0(*(_QWORD *)(v15[2] + 8), *(_QWORD *)(*v15 + 8));
        v15 = (__int64 *)a1[13];
        v16 = v43;
        if ( v38 )
        {
          v12 = (__int64 *)v15[1];
          v39 = *(_QWORD *)(*v15 + 8);
          v17 = a1[14] - (_QWORD)v15;
          if ( *(__int64 **)(v39 + 56) == v12 && !*(_BYTE *)(v39 + 216) )
          {
            v47[0] = 0;
            v19 = v17 >> 4;
            v20 = 0;
            goto LABEL_17;
          }
        }
        else
        {
          v17 = a1[14] - (_QWORD)v15;
        }
      }
    }
    v19 = v17 >> 4;
    if ( !(_DWORD)v19 )
    {
LABEL_39:
      v20 = v47[0];
      goto LABEL_17;
    }
    v12 = v15 + 1;
    v20 = 0;
    while ( 1 )
    {
      v13 = (unsigned int)v20;
      v11 = *(_QWORD *)(*(v12 - 1) + 8);
      if ( v18 == v11 || *(_BYTE *)(v11 + 216) )
        break;
      if ( v46 == v11 )
        goto LABEL_16;
      v13 = *v12;
      if ( *(_QWORD *)(v11 + 56) == *v12 )
        v47[0] = v20;
LABEL_11:
      ++v20;
      v12 += 2;
      if ( (unsigned int)v19 == v20 )
        goto LABEL_39;
    }
    if ( *(_QWORD *)(v11 + 56) == *v12 || v46 != v11 )
      goto LABEL_11;
LABEL_16:
    v47[0] = v20;
LABEL_17:
    if ( v19 == v20 || (v21 = &v15[2 * v20], v22 = *(_QWORD *)(*v21 + 8), v46 == v22) && *(_QWORD *)(v22 + 56) != v21[1] )
    {
      if ( (unsigned __int8)sub_34C41B0((__int64)a1, &v46, a2, (unsigned int)v16, v47) )
      {
        v22 = *(_QWORD *)(*(_QWORD *)(a1[13] + 16LL * v47[0]) + 8LL);
        goto LABEL_22;
      }
LABEL_40:
      sub_34BF650(a1, v8, a2, v46, v9);
    }
    else
    {
LABEL_22:
      sub_34C0410((__int64)a1, v22, v11, (__int64)v12, v13, v16);
      sub_34BF740((__int64)a1, v47[0], v23, v24, v25, v26);
      v45 = (unsigned int)((a1[14] - a1[13]) >> 4);
      v27 = 0;
      if ( (unsigned int)((a1[14] - a1[13]) >> 4) )
      {
        do
        {
          if ( v47[0] != (_DWORD)v27 )
          {
            sub_34BF360((__int64)a1, *(_QWORD **)(a1[13] + 16 * v27 + 8), v22);
            v28 = *(_QWORD *)(a1[13] + 16 * v27);
            v29 = a1[1];
            v30 = v28 + 24;
            if ( v29 != v28 + 24 )
            {
              v31 = 0xAAAAAAAAAAAAAAABLL * ((v29 - v30) >> 3);
              if ( v29 - v30 <= 0 )
              {
                v30 = a1[1];
              }
              else
              {
                v32 = (_QWORD *)(v28 + 40);
                do
                {
                  v33 = *(v32 - 3);
                  *((_DWORD *)v32 - 10) = *((_DWORD *)v32 - 4);
                  *(v32 - 4) = *(v32 - 1);
                  if ( v33 )
                    sub_B91220((__int64)(v32 - 3), v33);
                  v34 = (unsigned __int8 *)*v32;
                  *(v32 - 3) = *v32;
                  if ( v34 )
                  {
                    sub_B976B0((__int64)v32, v34, (__int64)(v32 - 3));
                    *v32 = 0;
                  }
                  v32 += 3;
                  --v31;
                }
                while ( v31 );
                v30 = a1[1];
              }
            }
            a1[1] = v30 - 24;
            v35 = *(_QWORD *)(v30 - 8);
            if ( v35 )
              sub_B91220(v30 - 8, v35);
          }
          ++v27;
        }
        while ( v45 != v27 );
      }
      v44 = 1;
    }
    v5 = a1[1];
  }
  while ( (unsigned __int64)(v5 - *a1) > 0x18 );
  return v44;
}
