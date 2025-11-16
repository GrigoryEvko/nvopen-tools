// Function: sub_1487810
// Address: 0x1487810
//
__int64 *__fastcall sub_1487810(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r13
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // ebx
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned int v14; // eax
  unsigned int v17; // ecx
  unsigned int v18; // eax
  __int64 v19; // rbx
  __int64 v20; // rbx
  unsigned int v21; // eax
  unsigned int v22; // ebx
  unsigned int v23; // eax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v35; // [rsp+38h] [rbp-B8h]
  __int64 *v36; // [rsp+40h] [rbp-B0h]
  unsigned int v37; // [rsp+48h] [rbp-A8h]
  __int64 v38; // [rsp+50h] [rbp-A0h]
  __int64 v39; // [rsp+50h] [rbp-A0h]
  int v40; // [rsp+58h] [rbp-98h]
  unsigned int v41; // [rsp+5Ch] [rbp-94h]
  unsigned __int64 v42; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-88h]
  __int64 v44; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-78h]
  __int64 v46; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v47; // [rsp+88h] [rbp-68h]
  unsigned __int64 v48; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+98h] [rbp-58h]
  unsigned __int64 v50; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v51; // [rsp+A8h] [rbp-48h]
  __int64 v52; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v53; // [rsp+B8h] [rbp-38h]

  v5 = **(_QWORD **)(a1 + 32);
  v40 = *(_QWORD *)(a1 + 40);
  if ( v40 != 1 )
  {
    v36 = **(__int64 ***)(a1 + 32);
    v41 = 1;
    while ( 1 )
    {
      v9 = sub_1456040(v5);
      v35 = v9;
      if ( v41 == 1 )
        break;
      if ( v41 > 0x3E8 )
      {
        v5 = sub_1456E90((__int64)a3);
        goto LABEL_4;
      }
      v37 = sub_1456C90((__int64)a3, v9);
      v43 = v37;
      if ( v37 > 0x40 )
      {
        sub_16A4EF0(&v42, 1, 0);
        v10 = v41;
        if ( v41 <= 2 )
          goto LABEL_52;
LABEL_15:
        v11 = 1;
        v12 = v10 + 1;
        v13 = 3;
        v38 = v12;
        while ( 2 )
        {
          LODWORD(v51) = v37;
          if ( v37 <= 0x40 )
          {
            v14 = v37;
            _RDX = v13 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v37);
          }
          else
          {
            sub_16A4EF0(&v50, v13, 0);
            v14 = v51;
            if ( (unsigned int)v51 > 0x40 )
            {
              v18 = sub_16A58A0(&v50);
              v11 += v18;
              sub_16A8110(&v50, v18);
              goto LABEL_22;
            }
            _RDX = v50;
          }
          if ( _RDX )
          {
            __asm { tzcnt   rsi, rdx }
            v17 = _RSI;
            if ( v14 <= (unsigned int)_RSI )
              v17 = v14;
            v11 += v17;
            if ( v14 > (unsigned int)_RSI )
            {
              v50 = _RDX >> v17;
LABEL_22:
              sub_16A7C10(&v42, &v50);
              if ( (unsigned int)v51 > 0x40 && v50 )
                j_j___libc_free_0_0(v50);
              if ( ++v13 == v38 )
              {
                v19 = 1LL << v11;
                goto LABEL_32;
              }
              continue;
            }
          }
          else
          {
            v11 += v14;
          }
          break;
        }
        v50 = 0;
        goto LABEL_22;
      }
      v10 = v41;
      v42 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v37) & 1;
      if ( v41 > 2 )
        goto LABEL_15;
LABEL_52:
      v19 = 2;
      v11 = 1;
LABEL_32:
      v45 = v11 + v37;
      if ( v11 + v37 <= 0x40 )
      {
        v44 = 0;
LABEL_34:
        v44 |= v19;
        goto LABEL_35;
      }
      sub_16A4EF0(&v44, 0, 0);
      if ( v45 <= 0x40 )
        goto LABEL_34;
      *(_QWORD *)(v44 + 8LL * (v11 >> 6)) |= v19;
LABEL_35:
      v47 = v37 + 1;
      v20 = 1LL << v37;
      if ( v37 + 1 > 0x40 )
      {
        sub_16A4EF0(&v46, 0, 0);
        if ( v47 > 0x40 )
        {
          *(_QWORD *)(v46 + 8LL * (v37 >> 6)) |= v20;
          goto LABEL_38;
        }
      }
      else
      {
        v46 = 0;
      }
      v46 |= v20;
LABEL_38:
      sub_16A5C50(&v48, &v42, v37 + 1);
      sub_16AE1A0(&v50, &v48, &v46);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      v48 = v50;
      v21 = v51;
      LODWORD(v51) = 0;
      v49 = v21;
      sub_135E100((__int64 *)&v50);
      sub_16A5A50(&v50, &v48);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      v22 = 1;
      v48 = v50;
      v23 = v51;
      LODWORD(v51) = 0;
      v49 = v23;
      sub_135E100((__int64 *)&v50);
      v24 = sub_15E0530(a3[3]);
      v39 = sub_1644900(v24, v11 + v37);
      v25 = sub_1483B20(a3, a2, v39, a4, a5);
      do
      {
        v26 = sub_1456040(a2);
        v27 = sub_145CF80((__int64)a3, v26, v22, 0);
        v28 = sub_14806B0((__int64)a3, a2, v27, 0, 0);
        v53 = sub_1483B20(a3, v28, v39, a4, a5);
        v52 = v25;
        v50 = (unsigned __int64)&v52;
        v51 = 0x200000002LL;
        v25 = sub_147EE30(a3, (__int64 **)&v50, 0, 0, a4, a5);
        if ( (__int64 *)v50 != &v52 )
          _libc_free(v50);
        ++v22;
      }
      while ( v22 != v41 );
      v29 = sub_145CF40((__int64)a3, (__int64)&v44);
      v30 = sub_1483CF0(a3, v25, v29, a4, a5);
      v31 = sub_1483B20(a3, v30, v35, a4, a5);
      v32 = sub_145CF40((__int64)a3, (__int64)&v48);
      v5 = sub_13A5B60((__int64)a3, v32, v31, 0, 0);
      sub_135E100((__int64 *)&v48);
      sub_135E100(&v46);
      sub_135E100(&v44);
      sub_135E100((__int64 *)&v42);
      if ( sub_14562D0(v5) )
        return (__int64 *)v5;
LABEL_5:
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * v41);
      v50 = (unsigned __int64)&v52;
      v53 = v5;
      v52 = v7;
      v51 = 0x200000002LL;
      v8 = sub_147EE30(a3, (__int64 **)&v50, 0, 0, a4, a5);
      if ( (__int64 *)v50 != &v52 )
        _libc_free(v50);
      v50 = (unsigned __int64)&v52;
      v52 = (__int64)v36;
      v53 = v8;
      v51 = 0x200000002LL;
      v36 = sub_147DD40((__int64)a3, (__int64 *)&v50, 0, 0, a4, a5);
      if ( (__int64 *)v50 != &v52 )
        _libc_free(v50);
      if ( v40 == ++v41 )
        return v36;
      v5 = **(_QWORD **)(a1 + 32);
    }
    v5 = sub_1483B20(a3, a2, v9, a4, a5);
LABEL_4:
    if ( sub_14562D0(v5) )
      return (__int64 *)v5;
    goto LABEL_5;
  }
  return (__int64 *)v5;
}
