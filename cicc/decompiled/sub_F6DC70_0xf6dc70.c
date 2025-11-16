// Function: sub_F6DC70
// Address: 0xf6dc70
//
void __fastcall sub_F6DC70(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  const char *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned __int8 v12; // al
  bool v13; // dl
  __int64 v14; // rbx
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // r13
  unsigned __int8 v18; // al
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 *v21; // rdx
  const void *v22; // rax
  __int64 v23; // rdx
  unsigned __int8 v24; // al
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rdx
  size_t v32; // rbx
  __int64 *v33; // r13
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 *v41; // rax
  __m128i *v42; // r12
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // [rsp+8h] [rbp-98h]
  _BYTE *v48; // [rsp+18h] [rbp-88h]
  size_t n; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+28h] [rbp-78h]
  __int64 v51[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v52; // [rsp+40h] [rbp-60h] BYREF
  __int64 v53; // [rsp+48h] [rbp-58h]
  _QWORD v54[10]; // [rsp+50h] [rbp-50h] BYREF

  v6 = a3;
  v7 = a2;
  v52 = v54;
  v53 = 0x400000001LL;
  v54[0] = 0;
  v8 = sub_D49300(a1, (__int64)a2, a3, a4, a5, a6);
  if ( v8 )
  {
    v11 = v8;
    v12 = *(_BYTE *)(v8 - 16);
    v13 = (v12 & 2) != 0;
    v14 = (v12 & 2) != 0 ? *(unsigned int *)(v11 - 24) : (*(_WORD *)(v11 - 16) >> 6) & 0xFu;
    if ( (unsigned int)v14 > 1 )
    {
      v15 = 8 * v14;
      v16 = 8;
      v47 = v6;
      while ( 1 )
      {
        if ( v13 )
        {
          v17 = *(_QWORD *)(*(_QWORD *)(v11 - 32) + v16);
          v18 = *(_BYTE *)(v17 - 16);
          if ( (v18 & 2) != 0 )
            goto LABEL_7;
        }
        else
        {
          v17 = *(_QWORD *)(v11 + -16 - 8LL * ((v12 >> 2) & 0xF) + v16);
          v18 = *(_BYTE *)(v17 - 16);
          if ( (v18 & 2) != 0 )
          {
LABEL_7:
            if ( *(_DWORD *)(v17 - 24) != 2 )
              goto LABEL_8;
            v21 = *(__int64 **)(v17 - 32);
            v50 = v17 - 16;
            goto LABEL_17;
          }
        }
        if ( ((*(_WORD *)(v17 - 16) >> 6) & 0xF) != 2 )
          goto LABEL_8;
        v50 = v17 - 16;
        a2 = (char *)(v17 - 16 - 8LL * ((v18 >> 2) & 0xF));
        v21 = (__int64 *)a2;
LABEL_17:
        if ( *(_BYTE *)*v21 )
          goto LABEL_8;
        if ( v7 )
        {
          v48 = (_BYTE *)*v21;
          n = strlen(v7);
          v22 = (const void *)sub_B91420((__int64)v48);
          v9 = n;
          if ( n != v23 || n && (a2 = (char *)v7, memcmp(v22, v7, n)) )
          {
LABEL_8:
            v19 = (unsigned int)v53;
            v20 = (unsigned int)v53 + 1LL;
            if ( v20 > HIDWORD(v53) )
            {
              a2 = (char *)v54;
              sub_C8D5F0((__int64)&v52, v54, v20, 8u, v9, v10);
              v19 = (unsigned int)v53;
            }
            v52[v19] = v17;
            LODWORD(v53) = v53 + 1;
            goto LABEL_11;
          }
          v24 = *(_BYTE *)(v17 - 16);
          if ( (v24 & 2) != 0 )
          {
LABEL_23:
            v25 = *(_QWORD *)(v17 - 32);
            goto LABEL_24;
          }
        }
        else
        {
          sub_B91420(*v21);
          if ( v31 )
            goto LABEL_8;
          v24 = *(_BYTE *)(v17 - 16);
          if ( (v24 & 2) != 0 )
            goto LABEL_23;
        }
        v25 = v50 - 8LL * ((v24 >> 2) & 0xF);
LABEL_24:
        v26 = *(_QWORD *)(v25 + 8);
        if ( v26 )
        {
          v27 = *(_QWORD *)(v26 + 136);
          if ( v27 )
          {
            v28 = *(__int64 **)(v27 + 24);
            v29 = *(_DWORD *)(v27 + 32);
            if ( v29 > 0x40 )
            {
              v30 = *v28;
            }
            else
            {
              v30 = 0;
              if ( v29 )
                v30 = (__int64)((_QWORD)v28 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
            }
            if ( v47 == v30 )
              goto LABEL_30;
          }
        }
LABEL_11:
        v16 += 8;
        if ( v16 == v15 )
          goto LABEL_39;
        v12 = *(_BYTE *)(v11 - 16);
        v13 = (v12 & 2) != 0;
      }
    }
  }
  v47 = v6;
LABEL_39:
  v32 = 0;
  if ( v7 )
    v32 = strlen(v7);
  v33 = (__int64 *)sub_AA48A0(**(_QWORD **)(a1 + 32));
  v51[0] = sub_B9B140(v33, v7, v32);
  v34 = sub_BCB2D0(v33);
  v35 = sub_ACD640(v34, v47, 0);
  v51[1] = (__int64)sub_B98A20(v35, v47);
  v36 = sub_B9C770(v33, v51, (__int64 *)2, 0, 1);
  v39 = (unsigned int)v53;
  v40 = (unsigned int)v53 + 1LL;
  if ( v40 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, v40, 8u, v37, v38);
    v39 = (unsigned int)v53;
  }
  v52[v39] = v36;
  LODWORD(v53) = v53 + 1;
  v41 = (__int64 *)sub_AA48A0(**(_QWORD **)(a1 + 32));
  v42 = (__m128i *)sub_B9C770(v41, v52, (__int64 *)(unsigned int)v53, 0, 1);
  sub_BA6610(v42, 0, (unsigned __int8 *)v42);
  a2 = (char *)v42;
  sub_D49440(a1, (__int64)v42, v43, v44, v45, v46);
LABEL_30:
  if ( v52 != v54 )
    _libc_free(v52, a2);
}
