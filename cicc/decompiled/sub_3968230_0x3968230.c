// Function: sub_3968230
// Address: 0x3968230
//
__int64 __fastcall sub_3968230(__int64 a1, char a2)
{
  unsigned int v3; // r13d
  unsigned __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rbx
  void *v9; // rax
  char *v10; // rax
  size_t v11; // rdx
  __int64 v12; // r8
  void *v13; // rdi
  void *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  void *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int v23; // edi
  __int64 *v24; // rsi
  __int64 v25; // r10
  __int64 v26; // rdi
  unsigned int v27; // r9d
  unsigned int v28; // edx
  unsigned int v29; // ecx
  __int64 v30; // rsi
  __int64 v31; // rax
  unsigned int v32; // esi
  unsigned int v33; // edx
  unsigned int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rbx
  _QWORD *v38; // r12
  unsigned __int64 v39; // r13
  int v40; // esi
  int v41; // ecx
  unsigned __int8 v42; // [rsp+0h] [rbp-A0h]
  __int64 v43; // [rsp+0h] [rbp-A0h]
  __int64 v45; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+18h] [rbp-88h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  size_t v49; // [rsp+18h] [rbp-88h]
  __int64 v50; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v51[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v52; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v53; // [rsp+58h] [rbp-48h]
  __int64 v54; // [rsp+60h] [rbp-40h]
  unsigned int v55; // [rsp+68h] [rbp-38h]

  sub_39659E0(v51, *(_QWORD *)a1);
  v3 = sub_3962BB0(v51, *(_QWORD *)(a1 + 8));
  if ( (_BYTE)v3 )
    goto LABEL_2;
  v5 = *(_QWORD *)(a1 + 120);
  ++*(_QWORD *)(a1 + 112);
  v6 = *(_DWORD *)(a1 + 136);
  v52 = 1;
  v53 = (_QWORD *)v5;
  v7 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  v55 = v6;
  v54 = v7;
  sub_1C2CBE0((__int64 *)a1);
  v42 = 1;
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v45 = *(_QWORD *)a1 + 72LL;
  while ( v45 != v8 )
  {
    v19 = *(_QWORD *)(a1 + 16);
    v20 = v8 - 24;
    if ( !v8 )
      v20 = 0;
    v21 = *(unsigned int *)(v19 + 48);
    v50 = v20;
    if ( (_DWORD)v21 )
    {
      v22 = *(_QWORD *)(v19 + 32);
      v23 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v24 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v24;
      if ( v20 != *v24 )
      {
        v40 = 1;
        while ( v25 != -8 )
        {
          v41 = v40 + 1;
          v23 = (v21 - 1) & (v40 + v23);
          v24 = (__int64 *)(v22 + 16LL * v23);
          v25 = *v24;
          if ( v20 == *v24 )
            goto LABEL_17;
          v40 = v41;
        }
        goto LABEL_12;
      }
LABEL_17:
      if ( v24 != (__int64 *)(v22 + 16 * v21) && v24[1] )
      {
        v48 = sub_3967E50(a1 + 112, &v50)[1];
        v26 = sub_3967E50((__int64)&v52, &v50)[1];
        if ( *(_DWORD *)(v26 + 8) == *(_DWORD *)(v48 + 8) && *(_DWORD *)(v26 + 12) == *(_DWORD *)(v48 + 12) )
        {
          v27 = (unsigned int)(*(_DWORD *)(v26 + 40) + 63) >> 6;
          v28 = (unsigned int)(*(_DWORD *)(v48 + 40) + 63) >> 6;
          v29 = v28;
          if ( v27 <= v28 )
            v28 = (unsigned int)(*(_DWORD *)(v26 + 40) + 63) >> 6;
          if ( v28 )
          {
            v30 = v28 + 1;
            v31 = 1;
            while ( *(_QWORD *)(*(_QWORD *)(v26 + 24) + 8 * v31 - 8) == *(_QWORD *)(*(_QWORD *)(v48 + 24) + 8 * v31 - 8) )
            {
              v28 = v31++;
              if ( v30 == v31 )
                goto LABEL_27;
            }
          }
          else
          {
LABEL_27:
            if ( v27 == v28 )
            {
              if ( v29 == v28 )
              {
LABEL_30:
                v32 = (unsigned int)(*(_DWORD *)(v26 + 64) + 63) >> 6;
                v33 = (unsigned int)(*(_DWORD *)(v48 + 64) + 63) >> 6;
                v34 = v33;
                if ( v32 <= v33 )
                  v33 = (unsigned int)(*(_DWORD *)(v26 + 64) + 63) >> 6;
                if ( v33 )
                {
                  v35 = 0;
                  while ( *(_QWORD *)(*(_QWORD *)(v26 + 48) + 8 * v35) == *(_QWORD *)(*(_QWORD *)(v48 + 48) + 8 * v35) )
                  {
                    if ( ++v35 == v33 )
                      goto LABEL_36;
                  }
                }
                else
                {
LABEL_36:
                  if ( v32 == v33 )
                  {
                    if ( v34 == v33 )
                      goto LABEL_12;
                    while ( !*(_QWORD *)(*(_QWORD *)(v48 + 48) + 8LL * v33) )
                    {
                      if ( v34 == ++v33 )
                        goto LABEL_12;
                    }
                  }
                  else
                  {
                    v36 = *(_QWORD *)(v26 + 48);
                    while ( !*(_QWORD *)(v36 + 8LL * v33) )
                    {
                      if ( v32 == ++v33 )
                        goto LABEL_12;
                    }
                  }
                }
              }
              else
              {
                while ( !*(_QWORD *)(*(_QWORD *)(v48 + 24) + 8LL * v28) )
                {
                  if ( v29 == ++v28 )
                    goto LABEL_30;
                }
              }
            }
            else
            {
              while ( !*(_QWORD *)(*(_QWORD *)(v26 + 24) + 8LL * v28) )
              {
                if ( v27 == ++v28 )
                  goto LABEL_30;
              }
            }
          }
        }
        if ( !a2 )
        {
          v42 = 0;
          break;
        }
        v9 = sub_16E8CB0();
        v46 = sub_1263B40((__int64)v9, "BB: ");
        v10 = (char *)sub_1649960(v50);
        v12 = v46;
        v13 = *(void **)(v46 + 24);
        if ( *(_QWORD *)(v46 + 16) - (_QWORD)v13 < v11 )
        {
          v12 = sub_16E7EE0(v46, v10, v11);
        }
        else if ( v11 )
        {
          v43 = v46;
          v49 = v11;
          memcpy(v13, v10, v11);
          v12 = v43;
          *(_QWORD *)(v43 + 24) += v49;
        }
        sub_1263B40(v12, "\n");
        v47 = sub_3967E50(a1 + 112, &v50)[1];
        v14 = sub_16E8CB0();
        v15 = sub_1263B40((__int64)v14, "Correct RP Info\n");
        sub_3962EB0(a1, v15, v47);
        v16 = sub_3967E50((__int64)&v52, &v50)[1];
        v17 = sub_16E8CB0();
        v18 = sub_1263B40((__int64)v17, "Incorrect RP Info\n");
        sub_3962EB0(a1, v18, v16);
        v42 = 0;
      }
    }
LABEL_12:
    v8 = *(_QWORD *)(v8 + 8);
  }
  if ( v55 )
  {
    v37 = v53;
    v38 = &v53[2 * v55];
    do
    {
      if ( *v37 != -8 && *v37 != -16 )
      {
        v39 = v37[1];
        if ( v39 )
        {
          _libc_free(*(_QWORD *)(v39 + 48));
          _libc_free(*(_QWORD *)(v39 + 24));
          j_j___libc_free_0(v39);
        }
      }
      v37 += 2;
    }
    while ( v38 != v37 );
  }
  j___libc_free_0((unsigned __int64)v53);
  v3 = v42;
LABEL_2:
  if ( v51[0] )
    j_j___libc_free_0((unsigned __int64)v51[0]);
  return v3;
}
