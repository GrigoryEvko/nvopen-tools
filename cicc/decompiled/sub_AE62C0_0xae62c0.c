// Function: sub_AE62C0
// Address: 0xae62c0
//
void __fastcall sub_AE62C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v5; // rax
  __int64 *v7; // rsi
  _BYTE *v8; // rbx
  __int64 *v9; // rdi
  __int64 **v10; // rbx
  __int64 *v11; // r13
  __int64 *v12; // r9
  __int64 *v13; // r14
  __int64 v14; // rbx
  __int64 **v15; // r15
  __int64 *v16; // r12
  __int64 *v17; // rax
  __int64 *v18; // rdx
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int64 *v23; // [rsp-1A0h] [rbp-1A0h]
  __int64 **v24; // [rsp-180h] [rbp-180h]
  __int64 *v25; // [rsp-170h] [rbp-170h]
  _QWORD v26[6]; // [rsp-168h] [rbp-168h] BYREF
  __int64 v27; // [rsp-138h] [rbp-138h] BYREF
  __int64 *v28; // [rsp-130h] [rbp-130h]
  __int64 v29; // [rsp-128h] [rbp-128h]
  int v30; // [rsp-120h] [rbp-120h]
  char v31; // [rsp-11Ch] [rbp-11Ch]
  __int64 v32; // [rsp-118h] [rbp-118h] BYREF
  __int64 v33; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 *v34; // [rsp-F0h] [rbp-F0h]
  __int64 v35; // [rsp-E8h] [rbp-E8h]
  int v36; // [rsp-E0h] [rbp-E0h]
  char v37; // [rsp-DCh] [rbp-DCh]
  __int64 v38; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 *v39; // [rsp-B8h] [rbp-B8h] BYREF
  int v40; // [rsp-B0h] [rbp-B0h]
  __int64 v41; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 *v42; // [rsp-78h] [rbp-78h] BYREF
  int v43; // [rsp-70h] [rbp-70h]
  __int64 v44; // [rsp-68h] [rbp-68h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 8) != 0 )
  {
    v3 = a3;
    v5 = sub_BD5C60(a2, a2, a3);
    v7 = &v33;
    v28 = &v32;
    v26[3] = a1;
    v27 = 0;
    v29 = 4;
    v30 = 0;
    v31 = 1;
    v33 = 0;
    v34 = &v38;
    v35 = 4;
    v36 = 0;
    v37 = 1;
    v26[0] = v5;
    v26[1] = &v27;
    v26[2] = &v33;
    v26[4] = v3;
    v8 = (_BYTE *)sub_B91390(a2);
    if ( v8 )
    {
      sub_AE6050(v26, v8);
      v7 = (__int64 *)(v8 + 8);
      sub_B962A0(&v39, v8 + 8);
      v9 = v39;
      v25 = &v39[v40];
      if ( v25 != v39 )
      {
        v10 = (__int64 **)v39;
        v23 = (__int64 *)(v3 + 16);
LABEL_6:
        while ( 1 )
        {
          v11 = *v10;
          v7 = *v10;
          sub_AE6050(v26, *v10);
          if ( v3 )
            break;
LABEL_5:
          if ( v25 == (__int64 *)++v10 )
            goto LABEL_18;
        }
        v7 = v11 + 1;
        sub_B967C0(&v42, v11 + 1);
        v12 = v42;
        v13 = &v42[v43];
        if ( v13 == v42 )
          goto LABEL_16;
        v24 = v10;
        v14 = v3;
        v15 = (__int64 **)v42;
        while ( 1 )
        {
          v16 = *v15;
          if ( !v37 )
            goto LABEL_24;
          v17 = v34;
          v18 = &v34[HIDWORD(v35)];
          if ( v34 != v18 )
          {
            while ( v16 != (__int64 *)*v17 )
            {
              if ( v18 == ++v17 )
                goto LABEL_28;
            }
            goto LABEL_14;
          }
LABEL_28:
          if ( HIDWORD(v35) < (unsigned int)v35 )
          {
            ++HIDWORD(v35);
            *v18 = (__int64)v16;
            v20 = *(unsigned int *)(v14 + 8);
            v22 = *(unsigned int *)(v14 + 12);
            ++v33;
            v21 = v20 + 1;
            if ( v20 + 1 > v22 )
            {
LABEL_30:
              v7 = v23;
              sub_C8D5F0(v14, v23, v21, 8);
              v20 = *(unsigned int *)(v14 + 8);
            }
LABEL_26:
            ++v15;
            *(_QWORD *)(*(_QWORD *)v14 + 8 * v20) = v16;
            ++*(_DWORD *)(v14 + 8);
            if ( v13 == (__int64 *)v15 )
            {
LABEL_15:
              v3 = v14;
              v12 = v42;
              v10 = v24;
LABEL_16:
              if ( v12 == &v44 )
                goto LABEL_5;
              ++v10;
              _libc_free(v12, v7);
              if ( v25 == (__int64 *)v10 )
              {
LABEL_18:
                v9 = v39;
                break;
              }
              goto LABEL_6;
            }
          }
          else
          {
LABEL_24:
            v7 = *v15;
            sub_C8CC70(&v33, *v15);
            if ( v19 )
            {
              v20 = *(unsigned int *)(v14 + 8);
              v21 = v20 + 1;
              if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v14 + 12) )
                goto LABEL_30;
              goto LABEL_26;
            }
LABEL_14:
            if ( v13 == (__int64 *)++v15 )
              goto LABEL_15;
          }
        }
      }
      if ( v9 != &v41 )
        _libc_free(v9, v7);
    }
    if ( v37 )
    {
      if ( v31 )
        return;
LABEL_32:
      _libc_free(v28, v7);
      return;
    }
    _libc_free(v34, v7);
    if ( !v31 )
      goto LABEL_32;
  }
}
