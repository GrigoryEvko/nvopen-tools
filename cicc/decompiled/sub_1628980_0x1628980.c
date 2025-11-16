// Function: sub_1628980
// Address: 0x1628980
//
void __fastcall sub_1628980(__int64 a1, __int64 a2, unsigned int a3)
{
  _BYTE *v3; // r12
  _BYTE *v4; // r15
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // r8
  void *v8; // r9
  _QWORD *v9; // r14
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r8
  size_t v15; // rdx
  __int64 v16; // r13
  unsigned int v17; // eax
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  size_t v22; // [rsp+8h] [rbp-108h]
  void *src; // [rsp+10h] [rbp-100h]
  __int64 v24; // [rsp+18h] [rbp-F8h]
  __int64 n; // [rsp+20h] [rbp-F0h]
  __int64 *na; // [rsp+20h] [rbp-F0h]
  size_t nb; // [rsp+20h] [rbp-F0h]
  size_t nc; // [rsp+20h] [rbp-F0h]
  size_t nd; // [rsp+20h] [rbp-F0h]
  __int64 v31; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-D8h]
  _QWORD *v33; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+48h] [rbp-C8h]
  _BYTE *v35; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-B8h]
  _BYTE v37[176]; // [rsp+60h] [rbp-B0h] BYREF

  v35 = v37;
  v36 = 0x800000000LL;
  sub_1626D60(a2, (__int64)&v35);
  v3 = &v35[16 * (unsigned int)v36];
  if ( v35 == v3 )
    goto LABEL_14;
  v4 = v35;
  do
  {
    v5 = *(_DWORD *)v4;
    v6 = *((_QWORD *)v4 + 1);
    if ( !a3 )
      goto LABEL_11;
    if ( v5 != 19 )
    {
      if ( !v5 )
      {
        if ( *(_BYTE *)v6 == 24
          || (v12 = *(unsigned int *)(v6 + 8),
              v6 = *(_QWORD *)(v6 - 8 * v12),
              (v13 = *(_QWORD *)(*((_QWORD *)v4 + 1) + 8 * (1 - v12))) == 0) )
        {
          n = 16;
          v7 = 0;
          v8 = 0;
        }
        else
        {
          v8 = *(void **)(v13 + 24);
          v7 = *(_QWORD *)(v13 + 32) - (_QWORD)v8;
          if ( (unsigned __int64)((v7 >> 3) + 2) > 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
          if ( v7 >> 3 == -2 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          n = v7 + 16;
        }
        v22 = v7;
        src = v8;
        v9 = (_QWORD *)sub_22077B0(n);
        memset(v9, 0, n);
        *v9 = 35;
        v9[1] = a3;
        if ( v22 )
          memmove(v9 + 2, src, v22);
        v10 = (__int64 *)sub_16498A0(a1);
        v24 = sub_15C4420(v10, v9, n >> 3, 0, 1);
        v11 = (__int64 *)sub_16498A0(a1);
        v6 = sub_15C5570(v11, v6, v24, 0, 1);
        j_j___libc_free_0(v9, n);
        v5 = *(_DWORD *)v4;
      }
LABEL_11:
      sub_16267C0(a1, v5, v6);
      goto LABEL_12;
    }
    v14 = a3;
    v15 = *(_QWORD *)(*(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8)) + 136LL);
    v16 = *(_QWORD *)(v6 + 8 * (1LL - *(unsigned int *)(v6 + 8)));
    v32 = *(_DWORD *)(v15 + 32);
    if ( v32 > 0x40 )
    {
      nd = v15;
      sub_16A4FD0(&v31, v15 + 24);
      v14 = a3;
      v15 = nd;
    }
    else
    {
      v31 = *(_QWORD *)(v15 + 24);
    }
    na = (__int64 *)v15;
    sub_16A7490(&v31, v14);
    v17 = v32;
    v32 = 0;
    LODWORD(v34) = v17;
    v33 = (_QWORD *)v31;
    v18 = sub_15A1070(*na, (__int64)&v33);
    v19 = sub_1624210(v18);
    if ( (unsigned int)v34 > 0x40 && v33 )
    {
      nb = (size_t)v19;
      j_j___libc_free_0_0(v33);
      v19 = (_QWORD *)nb;
    }
    if ( v32 > 0x40 && v31 )
    {
      nc = (size_t)v19;
      j_j___libc_free_0_0(v31);
      v19 = (_QWORD *)nc;
    }
    v33 = v19;
    v34 = v16;
    v20 = (__int64 *)sub_16498A0(a1);
    v21 = sub_1627350(v20, (__int64 *)&v33, (__int64 *)2, 0, 1);
    sub_16267C0(a1, 0x13u, v21);
LABEL_12:
    v4 += 16;
  }
  while ( v3 != v4 );
  v3 = v35;
LABEL_14:
  if ( v3 != v37 )
    _libc_free((unsigned __int64)v3);
}
