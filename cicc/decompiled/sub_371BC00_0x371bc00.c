// Function: sub_371BC00
// Address: 0x371bc00
//
void __fastcall sub_371BC00(__int64 a1, __int64 a2, __int64 a3)
{
  char *v3; // r8
  size_t v4; // r15
  __int64 v5; // r9
  char *v7; // rbx
  unsigned __int64 v8; // rax
  _BYTE *v9; // rsi
  _BYTE *v10; // rdi
  __int64 v11; // rdx
  __int64 **v12; // rax
  __int64 v13; // r12
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  _BYTE *v20; // rdi
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  int v23; // eax
  char *v24; // [rsp+8h] [rbp-88h]
  char *v25; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+10h] [rbp-80h]
  int v27; // [rsp+10h] [rbp-80h]
  char *v28; // [rsp+10h] [rbp-80h]
  int v29; // [rsp+10h] [rbp-80h]
  char *v30; // [rsp+10h] [rbp-80h]
  int v31; // [rsp+18h] [rbp-78h]
  char *v32; // [rsp+18h] [rbp-78h]
  char *v33; // [rsp+18h] [rbp-78h]
  char *v34; // [rsp+18h] [rbp-78h]
  int v35; // [rsp+18h] [rbp-78h]
  int v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  int v38; // [rsp+18h] [rbp-78h]
  _BYTE *v39; // [rsp+20h] [rbp-70h] BYREF
  __int64 v40; // [rsp+28h] [rbp-68h]
  _BYTE src[96]; // [rsp+30h] [rbp-60h] BYREF

  v3 = (char *)a2;
  v4 = 8 * a3;
  v5 = (8 * a3) >> 3;
  LODWORD(a3) = v5;
  v7 = (char *)a2;
  v39 = src;
  v40 = 0x600000000LL;
  if ( v4 > 0x30 )
  {
    v35 = v5;
    sub_C8D5F0((__int64)&v39, src, v5, 8u, a2, v5);
    LODWORD(v5) = v35;
    v3 = (char *)a2;
    v20 = &v39[8 * (unsigned int)v40];
  }
  else
  {
    if ( !v4 )
    {
      LODWORD(v40) = v5;
      v5 = (unsigned int)v5;
LABEL_4:
      v8 = *(unsigned int *)(a1 + 56);
      a3 = (unsigned int)a3;
      if ( v8 >= (unsigned int)a3 )
      {
        v10 = src;
        if ( (_DWORD)a3 )
        {
          v28 = v3;
          v36 = v5;
          memmove(*(void **)(a1 + 48), src, 8LL * (unsigned int)a3);
          v10 = v39;
          v3 = v28;
          LODWORD(v5) = v36;
        }
      }
      else
      {
        if ( *(unsigned int *)(a1 + 60) < (unsigned __int64)(unsigned int)a3 )
        {
          *(_DWORD *)(a1 + 56) = 0;
          v30 = v3;
          v38 = v5;
          sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), (unsigned int)a3, 8u, (__int64)v3, v5);
          v10 = v39;
          a3 = (unsigned int)v40;
          v8 = 0;
          LODWORD(v5) = v38;
          v3 = v30;
          v9 = v39;
        }
        else
        {
          v9 = src;
          v10 = src;
          if ( *(_DWORD *)(a1 + 56) )
          {
            v24 = v3;
            v29 = v5;
            v37 = 8 * v8;
            memmove(*(void **)(a1 + 48), src, 8 * v8);
            v10 = v39;
            a3 = (unsigned int)v40;
            v3 = v24;
            LODWORD(v5) = v29;
            v9 = &v39[v37];
            v8 = v37;
          }
        }
        v11 = 8 * a3;
        if ( v9 != &v10[v11] )
        {
          v25 = v3;
          v31 = v5;
          memcpy((void *)(v8 + *(_QWORD *)(a1 + 48)), v9, v11 - v8);
          v10 = v39;
          v3 = v25;
          LODWORD(v5) = v31;
        }
      }
      *(_DWORD *)(a1 + 56) = v5;
      if ( v10 != src )
      {
        v32 = v3;
        _libc_free((unsigned __int64)v10);
        v3 = v32;
      }
      goto LABEL_12;
    }
    v20 = src;
  }
  v27 = v5;
  v34 = v3;
  memcpy(v20, v3, v4);
  v21 = (__int64)v39;
  v3 = v34;
  LODWORD(a3) = v27 + v40;
  LODWORD(v40) = a3;
  v5 = (unsigned int)a3;
  if ( v39 == src )
    goto LABEL_4;
  v22 = *(_QWORD *)(a1 + 48);
  if ( v22 != a1 + 64 )
  {
    _libc_free(v22);
    v21 = (__int64)v39;
    LODWORD(v5) = v40;
    v3 = v34;
  }
  *(_QWORD *)(a1 + 48) = v21;
  v23 = HIDWORD(v40);
  *(_DWORD *)(a1 + 56) = v5;
  *(_DWORD *)(a1 + 60) = v23;
LABEL_12:
  v12 = *(__int64 ***)(a1 + 120);
  v13 = 0;
  v14 = *v12;
  v33 = &v3[v4];
  if ( v3 != &v3[v4] )
  {
    do
    {
      LODWORD(v40) = 32;
      v39 = (_BYTE *)v13;
      v15 = sub_ACCFD0(v14, (__int64)&v39);
      v16 = v15;
      if ( (unsigned int)v40 > 0x40 )
      {
        if ( v39 )
        {
          v26 = v15;
          j_j___libc_free_0_0((unsigned __int64)v39);
          v16 = v26;
        }
      }
      v17 = *(_QWORD *)v7;
      v7 += 8;
      ++v13;
      v18 = *(_QWORD *)(v17 + 16);
      v39 = sub_B98A20(v16, (__int64)&v39);
      v19 = sub_B9C770(v14, (__int64 *)&v39, (__int64 *)1, 0, 1);
      sub_B9A090(v18, "sandboxaux", 0xAu, v19);
    }
    while ( v33 != v7 );
  }
}
