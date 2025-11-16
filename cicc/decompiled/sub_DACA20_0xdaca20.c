// Function: sub_DACA20
// Address: 0xdaca20
//
char __fastcall sub_DACA20(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int64 v9; // r9
  __int64 v10; // r8
  _BYTE *v11; // rsi
  __int64 *v12; // rdx
  __int16 v13; // ax
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  _QWORD *v18; // rax
  int v19; // eax
  _BYTE *v20; // r8
  __int64 v21; // rcx
  __int64 v22; // rdi
  unsigned __int16 v23; // dx
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // rbx
  __int64 *v30; // r15
  __int64 v31; // r14
  __int64 *v32; // rax
  __int16 v33; // ax
  __int64 *v34; // r10
  __int64 v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  _QWORD *v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // rax
  __int64 *v45; // [rsp+0h] [rbp-170h]
  __int64 *v46; // [rsp+0h] [rbp-170h]
  __int64 v47; // [rsp+8h] [rbp-168h]
  __int64 v48; // [rsp+8h] [rbp-168h]
  __int64 v49; // [rsp+8h] [rbp-168h]
  __int64 v50; // [rsp+20h] [rbp-150h] BYREF
  _BYTE *v51; // [rsp+28h] [rbp-148h] BYREF
  __int64 v52; // [rsp+30h] [rbp-140h]
  _BYTE v53[72]; // [rsp+38h] [rbp-138h] BYREF
  __int64 *v54; // [rsp+80h] [rbp-F0h]
  _BYTE *v55; // [rsp+88h] [rbp-E8h] BYREF
  __int64 v56; // [rsp+90h] [rbp-E0h]
  _BYTE v57[64]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v58; // [rsp+D8h] [rbp-98h] BYREF
  __int64 *v59; // [rsp+E0h] [rbp-90h]
  __int64 v60; // [rsp+E8h] [rbp-88h]
  int v61; // [rsp+F0h] [rbp-80h]
  char v62; // [rsp+F4h] [rbp-7Ch]
  __int64 v63; // [rsp+F8h] [rbp-78h] BYREF

  result = sub_D97040(a1, *(_QWORD *)(a3 + 8));
  if ( result )
  {
    v7 = sub_D98300(a1, a3);
    v10 = v7;
    if ( !v7 )
      return sub_DAC8D0(a1, (_BYTE *)a3);
    v62 = 1;
    v11 = v57;
    v12 = &v50;
    v51 = v53;
    v52 = 0x800000000LL;
    v56 = 0x800000000LL;
    v59 = &v63;
    v60 = 0x100000008LL;
    v61 = 0;
    v63 = v7;
    v58 = 1;
    v13 = *(_WORD *)(v7 + 24);
    v50 = a2;
    v54 = &v50;
    v55 = v57;
    if ( v13 == 15 )
    {
      v14 = *(_QWORD *)(v10 - 8);
      if ( *(_BYTE *)v14 > 0x1Cu )
      {
        v11 = *(_BYTE **)(v14 + 40);
        if ( *(_BYTE *)(a2 + 84) )
        {
          v15 = *(__int64 **)(a2 + 64);
          v12 = &v15[*(unsigned int *)(a2 + 76)];
          if ( v15 == v12 )
            goto LABEL_18;
          while ( v11 != (_BYTE *)*v15 )
          {
            if ( v12 == ++v15 )
              goto LABEL_18;
          }
        }
        else
        {
          v49 = v10;
          v44 = sub_C8CA60(a2 + 56, (__int64)v11);
          v10 = v49;
          if ( !v44 )
          {
            v16 = (unsigned int)v56;
            v17 = HIDWORD(v56);
            v9 = (unsigned int)v56 + 1LL;
LABEL_11:
            if ( v9 > v17 )
            {
              v11 = v57;
              v48 = v10;
              sub_C8D5F0((__int64)&v55, v57, v9, 8u, v10, v9);
              v16 = (unsigned int)v56;
              v10 = v48;
            }
LABEL_19:
            *(_QWORD *)&v55[8 * v16] = v10;
            v19 = v56 + 1;
            LODWORD(v56) = v56 + 1;
            while ( 1 )
            {
              v20 = v55;
              v21 = (__int64)&v55[8 * v19];
              if ( !v19 )
                break;
              while ( 1 )
              {
                v22 = *(_QWORD *)(v21 - 8);
                LODWORD(v56) = --v19;
                v23 = *(_WORD *)(v22 + 24);
                if ( v23 > 0xEu )
                {
                  if ( v23 != 15 )
                    BUG();
                  goto LABEL_23;
                }
                if ( v23 > 1u )
                  break;
LABEL_23:
                v21 -= 8;
                if ( !v19 )
                  goto LABEL_24;
              }
              v25 = sub_D960E0(v22);
              v29 = (__int64 *)(v25 + 8 * v26);
              v30 = (__int64 *)v25;
              if ( (__int64 *)v25 != v29 )
              {
                while ( 1 )
                {
                  v31 = *v30;
                  if ( v62 )
                  {
                    v32 = v59;
                    v27 = HIDWORD(v60);
                    v26 = (__int64)&v59[HIDWORD(v60)];
                    if ( v59 != (__int64 *)v26 )
                    {
                      while ( v31 != *v32 )
                      {
                        if ( (__int64 *)v26 == ++v32 )
                          goto LABEL_55;
                      }
                      goto LABEL_39;
                    }
LABEL_55:
                    if ( HIDWORD(v60) < (unsigned int)v60 )
                      break;
                  }
                  v11 = (_BYTE *)*v30;
                  sub_C8CC70((__int64)&v58, *v30, v26, v27, v28, v9);
                  if ( (_BYTE)v26 )
                    goto LABEL_42;
LABEL_39:
                  if ( v29 == ++v30 )
                    goto LABEL_40;
                }
                ++HIDWORD(v60);
                *(_QWORD *)v26 = v31;
                ++v58;
LABEL_42:
                v33 = *(_WORD *)(v31 + 24);
                v34 = v54;
                if ( v33 == 15 )
                {
                  v35 = *(_QWORD *)(v31 - 8);
                  if ( *(_BYTE *)v35 > 0x1Cu )
                  {
                    v36 = *v54;
                    v11 = *(_BYTE **)(v35 + 40);
                    if ( *(_BYTE *)(*v54 + 84) )
                    {
                      v37 = *(_QWORD **)(v36 + 64);
                      v38 = &v37[*(unsigned int *)(v36 + 76)];
                      if ( v37 == v38 )
                        goto LABEL_52;
                      while ( v11 != (_BYTE *)*v37 )
                      {
                        if ( v38 == ++v37 )
                          goto LABEL_52;
                      }
LABEL_49:
                      v39 = *((unsigned int *)v34 + 4);
                      if ( v39 + 1 > (unsigned __int64)*((unsigned int *)v34 + 5) )
                      {
                        v11 = v34 + 3;
                        v46 = v34;
                        sub_C8D5F0((__int64)(v34 + 1), v34 + 3, v39 + 1, 8u, v28, v9);
                        v34 = v46;
                        v39 = *((unsigned int *)v46 + 4);
                      }
                      *(_QWORD *)(v34[1] + 8 * v39) = v31;
                      ++*((_DWORD *)v34 + 4);
                      goto LABEL_52;
                    }
                    v45 = v54;
                    v43 = sub_C8CA60(v36 + 56, (__int64)v11);
                    v34 = v45;
                    if ( v43 )
                      goto LABEL_49;
                  }
                }
                else if ( v33 == 8 )
                {
                  v42 = *(_QWORD **)(v31 + 48);
                  if ( (_QWORD *)*v54 != v42 )
                  {
                    while ( v42 )
                    {
                      v42 = (_QWORD *)*v42;
                      if ( (_QWORD *)*v54 == v42 )
                        goto LABEL_49;
                    }
                    goto LABEL_52;
                  }
                  goto LABEL_49;
                }
LABEL_52:
                v40 = (unsigned int)v56;
                v27 = HIDWORD(v56);
                v41 = (unsigned int)v56 + 1LL;
                if ( v41 > HIDWORD(v56) )
                {
                  v11 = v57;
                  sub_C8D5F0((__int64)&v55, v57, v41, 8u, v28, v9);
                  v40 = (unsigned int)v56;
                }
                v26 = (__int64)v55;
                *(_QWORD *)&v55[8 * v40] = v31;
                LODWORD(v56) = v56 + 1;
                goto LABEL_39;
              }
LABEL_40:
              v19 = v56;
            }
LABEL_24:
            if ( !v62 )
            {
              _libc_free(v59, v11);
              v20 = v55;
            }
            if ( v20 != v57 )
              _libc_free(v20, v11);
            v24 = (__int64)v51;
            sub_DAB940(a1, (__int64)v51, (unsigned int)v52, v21, (__int64)v20, v9);
            if ( v51 != v53 )
              _libc_free(v51, v24);
            return sub_DAC8D0(a1, (_BYTE *)a3);
          }
        }
LABEL_10:
        v11 = (_BYTE *)v10;
        v47 = v10;
        sub_D9B3A0((__int64)&v51, v10, (__int64)v12, v8, v10, v9);
        v16 = (unsigned int)v56;
        v17 = HIDWORD(v56);
        v10 = v47;
        v9 = (unsigned int)v56 + 1LL;
        goto LABEL_11;
      }
    }
    else if ( v13 == 8 )
    {
      v18 = *(_QWORD **)(v10 + 48);
      if ( (_QWORD *)a2 != v18 )
      {
        while ( v18 )
        {
          v18 = (_QWORD *)*v18;
          if ( (_QWORD *)a2 == v18 )
            goto LABEL_10;
        }
        goto LABEL_18;
      }
      goto LABEL_10;
    }
LABEL_18:
    v16 = 0;
    goto LABEL_19;
  }
  return result;
}
