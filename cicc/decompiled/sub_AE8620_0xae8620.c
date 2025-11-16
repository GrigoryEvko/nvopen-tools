// Function: sub_AE8620
// Address: 0xae8620
//
void __fastcall sub_AE8620(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  bool v4; // al
  __int64 v5; // rsi
  __int64 v6; // rcx
  unsigned __int8 v7; // si
  __int64 *v8; // r12
  __int64 v9; // rcx
  __int64 *v10; // r15
  __int64 *v11; // r13
  __int64 v12; // r13
  unsigned __int8 v13; // al
  unsigned __int8 **v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // r13
  __int64 v17; // r13
  unsigned __int8 v18; // al
  __int64 v19; // rsi
  __int64 v20; // rcx
  unsigned __int8 v21; // si
  unsigned __int8 **v22; // r12
  __int64 v23; // rcx
  unsigned __int8 **v24; // r13
  unsigned __int8 *v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rcx
  unsigned __int8 v28; // si
  unsigned __int8 **v29; // r15
  __int64 v30; // rcx
  unsigned __int8 **v31; // r12
  __int64 v32; // r13
  unsigned __int8 *v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned __int8 v37; // dl
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 *v40; // r12
  __int64 v41; // r13
  __int64 v42; // rax
  unsigned __int8 *v43; // rsi
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int8 v46; // dl
  unsigned __int8 v47; // al
  __int64 v48; // rsi
  __int64 v49; // [rsp+0h] [rbp-40h]
  __int64 v50; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_AE7C90(a1, a2) )
  {
    v3 = *(_BYTE *)(a2 - 16);
    v49 = a2 - 16;
    v4 = (v3 & 2) != 0;
    if ( (v3 & 2) != 0 )
      v5 = *(_QWORD *)(a2 - 32);
    else
      v5 = v49 - 8LL * ((v3 >> 2) & 0xF);
    v6 = *(_QWORD *)(v5 + 48);
    if ( v6 )
    {
      v7 = *(_BYTE *)(v6 - 16);
      if ( (v7 & 2) != 0 )
      {
        v8 = *(__int64 **)(v6 - 32);
        v9 = *(unsigned int *)(v6 - 24);
      }
      else
      {
        v8 = (__int64 *)(v6 - 16 - 8LL * ((v7 >> 2) & 0xF));
        v9 = (*(_WORD *)(v6 - 16) >> 6) & 0xF;
      }
      v10 = &v8[v9];
      if ( v8 != v10 )
      {
        do
        {
          v17 = *v8;
          if ( (unsigned __int8)sub_AE7D80(a1, *v8) )
          {
            v18 = *(_BYTE *)(v17 - 16);
            if ( (v18 & 2) != 0 )
              v11 = *(__int64 **)(v17 - 32);
            else
              v11 = (__int64 *)(v17 - 16 - 8LL * ((v18 >> 2) & 0xF));
            v12 = *v11;
            v50 = v12 - 16;
            v13 = *(_BYTE *)(v12 - 16);
            if ( (v13 & 2) != 0 )
              v14 = *(unsigned __int8 ***)(v12 - 32);
            else
              v14 = (unsigned __int8 **)(v50 - 8LL * ((v13 >> 2) & 0xF));
            sub_AE8080(a1, *v14);
            v15 = *(_BYTE *)(v12 - 16);
            if ( (v15 & 2) != 0 )
              v16 = *(_QWORD *)(v12 - 32);
            else
              v16 = v50 - 8LL * ((v15 >> 2) & 0xF);
            sub_AE8230(a1, *(unsigned __int8 **)(v16 + 24));
          }
          ++v8;
        }
        while ( v10 != v8 );
        v3 = *(_BYTE *)(a2 - 16);
        v4 = (v3 & 2) != 0;
      }
    }
    if ( v4 )
      v19 = *(_QWORD *)(a2 - 32);
    else
      v19 = v49 - 8LL * ((v3 >> 2) & 0xF);
    v20 = *(_QWORD *)(v19 + 32);
    if ( v20 )
    {
      v21 = *(_BYTE *)(v20 - 16);
      if ( (v21 & 2) != 0 )
      {
        v22 = *(unsigned __int8 ***)(v20 - 32);
        v23 = *(unsigned int *)(v20 - 24);
      }
      else
      {
        v22 = (unsigned __int8 **)(v20 - 16 - 8LL * ((v21 >> 2) & 0xF));
        v23 = (*(_WORD *)(v20 - 16) >> 6) & 0xF;
      }
      v24 = &v22[v23];
      if ( v24 != v22 )
      {
        do
        {
          v25 = *v22++;
          sub_AE8230(a1, v25);
        }
        while ( v24 != v22 );
        v3 = *(_BYTE *)(a2 - 16);
        v4 = (v3 & 2) != 0;
      }
    }
    if ( v4 )
      v26 = *(_QWORD *)(a2 - 32);
    else
      v26 = v49 - 8LL * ((v3 >> 2) & 0xF);
    v27 = *(_QWORD *)(v26 + 40);
    if ( v27 )
    {
      v28 = *(_BYTE *)(v27 - 16);
      if ( (v28 & 2) != 0 )
      {
        v29 = *(unsigned __int8 ***)(v27 - 32);
        v30 = *(unsigned int *)(v27 - 24);
      }
      else
      {
        v29 = (unsigned __int8 **)(v27 - 16 - 8LL * ((v28 >> 2) & 0xF));
        v30 = (*(_WORD *)(v27 - 16) >> 6) & 0xF;
      }
      v31 = &v29[v30];
      if ( v31 != v29 )
      {
        v32 = 0x140000F000LL;
        do
        {
          while ( 1 )
          {
            v33 = *v29;
            v34 = **v29;
            if ( (unsigned __int8)v34 <= 0x24u )
            {
              if ( _bittest64(&v32, v34) )
                break;
            }
            sub_AE8440(a1, (__int64)v33);
            if ( v31 == ++v29 )
              goto LABEL_42;
          }
          sub_AE8230(a1, v33);
          ++v29;
        }
        while ( v31 != v29 );
LABEL_42:
        v3 = *(_BYTE *)(a2 - 16);
        v4 = (v3 & 2) != 0;
      }
    }
    v35 = v4 ? *(_QWORD *)(a2 - 32) : v49 - 8LL * ((v3 >> 2) & 0xF);
    v36 = *(_QWORD *)(v35 + 56);
    if ( v36 )
    {
      v37 = *(_BYTE *)(v36 - 16);
      if ( (v37 & 2) != 0 )
      {
        v38 = *(__int64 **)(v36 - 32);
        v39 = *(unsigned int *)(v36 - 24);
      }
      else
      {
        v38 = (__int64 *)(v36 - 16 - 8LL * ((v37 >> 2) & 0xF));
        v39 = (*(_WORD *)(v36 - 16) >> 6) & 0xF;
      }
      v40 = &v38[v39];
      if ( v40 != v38 )
      {
        v41 = 0x140000F000LL;
        do
        {
          v45 = *v38;
          v46 = *(_BYTE *)(*v38 - 16);
          if ( (v46 & 2) != 0 )
            v42 = *(_QWORD *)(v45 - 32);
          else
            v42 = v45 - 16 - 8LL * ((v46 >> 2) & 0xF);
          v43 = *(unsigned __int8 **)(v42 + 8);
          v44 = *v43;
          if ( (unsigned __int8)v44 <= 0x24u && _bittest64(&v41, v44) )
          {
            sub_AE8230(a1, v43);
          }
          else
          {
            if ( (_BYTE)v44 != 18 )
            {
              if ( (_BYTE)v44 == 21 )
              {
                v47 = *(v43 - 16);
                if ( (v47 & 2) != 0 )
                  goto LABEL_61;
LABEL_65:
                v48 = (__int64)&v43[-8 * ((v47 >> 2) & 0xF) - 16];
              }
              else
              {
                if ( (_BYTE)v44 != 22 )
                  goto LABEL_55;
                v47 = *(v43 - 16);
                if ( (v47 & 2) == 0 )
                  goto LABEL_65;
LABEL_61:
                v48 = *((_QWORD *)v43 - 4);
              }
              sub_AE8080(a1, *(unsigned __int8 **)(v48 + 8));
              goto LABEL_55;
            }
            sub_AE8440(a1, (__int64)v43);
          }
LABEL_55:
          ++v38;
        }
        while ( v40 != v38 );
      }
    }
  }
}
