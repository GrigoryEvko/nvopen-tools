// Function: sub_AE6870
// Address: 0xae6870
//
void __fastcall sub_AE6870(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rbx
  _BYTE *v4; // rax
  _BYTE *v5; // r12
  __int64 *v6; // r12
  _BYTE *v7; // r13
  __int64 *v8; // r9
  __int64 v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  char v17; // dl
  _BYTE *v18; // [rsp-198h] [rbp-198h]
  __int64 *v19; // [rsp-170h] [rbp-170h]
  _QWORD v20[6]; // [rsp-168h] [rbp-168h] BYREF
  __int64 v21; // [rsp-138h] [rbp-138h] BYREF
  __int64 *v22; // [rsp-130h] [rbp-130h]
  __int64 v23; // [rsp-128h] [rbp-128h]
  int v24; // [rsp-120h] [rbp-120h]
  char v25; // [rsp-11Ch] [rbp-11Ch]
  __int64 v26; // [rsp-118h] [rbp-118h] BYREF
  __int64 v27; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 *v28; // [rsp-F0h] [rbp-F0h]
  __int64 v29; // [rsp-E8h] [rbp-E8h]
  int v30; // [rsp-E0h] [rbp-E0h]
  char v31; // [rsp-DCh] [rbp-DCh]
  __int64 v32; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 *v33; // [rsp-B8h] [rbp-B8h] BYREF
  int v34; // [rsp-B0h] [rbp-B0h]
  __int64 v35; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 *v36; // [rsp-78h] [rbp-78h] BYREF
  int v37; // [rsp-70h] [rbp-70h]
  __int64 v38; // [rsp-68h] [rbp-68h] BYREF

  if ( (a2[7] & 8) != 0 )
  {
    v3 = a3;
    v21 = 0;
    v22 = &v26;
    v20[2] = &v27;
    v23 = 4;
    v24 = 0;
    v25 = 1;
    v27 = 0;
    v28 = &v32;
    v29 = 4;
    v30 = 0;
    v31 = 1;
    v20[0] = sub_BD5C60(a2, a2, a3);
    v20[1] = &v21;
    v20[3] = a1;
    v20[4] = v3;
    v4 = (_BYTE *)sub_B91390(a2);
    v5 = v4;
    if ( v4 )
    {
      sub_AE6600(v20, v4);
      a2 = v5 + 8;
      sub_B962A0(&v33, v5 + 8);
      v6 = v33;
      v19 = &v33[v34];
      if ( v19 != v33 )
      {
        v18 = (_BYTE *)(v3 + 16);
LABEL_6:
        while ( 1 )
        {
          v7 = (_BYTE *)*v6;
          a2 = (_BYTE *)*v6;
          sub_AE6600(v20, (_BYTE *)*v6);
          if ( v3 )
            break;
LABEL_5:
          if ( v19 == ++v6 )
            goto LABEL_23;
        }
        a2 = v7 + 8;
        sub_B967C0(&v36, v7 + 8);
        v8 = v36;
        if ( &v36[v37] == v36 )
          goto LABEL_21;
        v9 = v3;
        v10 = v36;
        v11 = &v36[v37];
        v12 = v9;
        while ( 1 )
        {
          v13 = *v10;
          if ( (unsigned __int8)(*(_BYTE *)(*v10 + 64) - 1) > 1u )
            goto LABEL_9;
          if ( !v31 )
            goto LABEL_31;
          v14 = v28;
          v15 = &v28[HIDWORD(v29)];
          if ( v28 != v15 )
          {
            while ( v13 != *v14 )
            {
              if ( v15 == ++v14 )
                goto LABEL_15;
            }
            goto LABEL_9;
          }
LABEL_15:
          if ( HIDWORD(v29) < (unsigned int)v29 )
          {
            ++HIDWORD(v29);
            *v15 = v13;
            ++v27;
LABEL_17:
            v16 = *(unsigned int *)(v12 + 8);
            if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
            {
              a2 = v18;
              sub_C8D5F0(v12, v18, v16 + 1, 8);
              v16 = *(unsigned int *)(v12 + 8);
            }
            ++v10;
            *(_QWORD *)(*(_QWORD *)v12 + 8 * v16) = v13;
            ++*(_DWORD *)(v12 + 8);
            if ( v11 == v10 )
            {
LABEL_20:
              v8 = v36;
              v3 = v12;
LABEL_21:
              if ( v8 == &v38 )
                goto LABEL_5;
              ++v6;
              _libc_free(v8, a2);
              if ( v19 == v6 )
              {
LABEL_23:
                v6 = v33;
                break;
              }
              goto LABEL_6;
            }
          }
          else
          {
LABEL_31:
            a2 = (_BYTE *)*v10;
            sub_C8CC70(&v27, *v10);
            if ( v17 )
              goto LABEL_17;
LABEL_9:
            if ( v11 == ++v10 )
              goto LABEL_20;
          }
        }
      }
      if ( v6 != &v35 )
        _libc_free(v6, a2);
    }
    if ( !v31 )
      _libc_free(v28, a2);
    if ( !v25 )
      _libc_free(v22, a2);
  }
}
