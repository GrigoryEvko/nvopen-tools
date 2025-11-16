// Function: sub_3798E20
// Address: 0x3798e20
//
unsigned __int8 *__fastcall sub_3798E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _OWORD *v5; // rax
  _OWORD *v7; // r9
  unsigned __int64 v9; // rbx
  _OWORD *i; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // edx
  int v20; // edi
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // r12
  unsigned __int8 *v25; // r12
  __int128 v27; // [rsp-10h] [rbp-100h]
  _OWORD *v28; // [rsp+0h] [rbp-F0h]
  __int64 v29; // [rsp+20h] [rbp-D0h] BYREF
  int v30; // [rsp+28h] [rbp-C8h]
  _OWORD *v31; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+38h] [rbp-B8h]
  _OWORD v33[11]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = v33;
  v7 = v33;
  v9 = *(unsigned int *)(a2 + 64);
  v31 = v33;
  v32 = 0x800000000LL;
  if ( v9 )
  {
    if ( v9 > 8 )
    {
      sub_C8D5F0((__int64)&v31, v33, v9, 0x10u, a5, (__int64)v33);
      v5 = &v31[(unsigned int)v32];
      for ( i = &v31[v9]; i != v5; ++v5 )
      {
LABEL_4:
        if ( v5 )
        {
          *(_QWORD *)v5 = 0;
          *((_DWORD *)v5 + 2) = 0;
        }
      }
    }
    else
    {
      i = &v33[v9];
      if ( i != v33 )
        goto LABEL_4;
    }
    v11 = *(unsigned int *)(a2 + 64);
    LODWORD(v32) = v9;
    if ( (_DWORD)v11 )
    {
      v12 = 0;
      v13 = 0;
      v14 = 40 * v11;
      do
      {
        v15 = *(_QWORD *)(a2 + 40);
        v16 = *(_QWORD *)(v15 + v13);
        v17 = *(_QWORD *)(v15 + v13 + 8);
        v13 += 40;
        v18 = sub_37946F0(a1, v16, v17);
        v20 = v19;
        v21 = v18;
        v22 = (unsigned __int64)v31;
        *(_QWORD *)&v31[v12] = v21;
        *(_DWORD *)(v22 + v12 * 16 + 8) = v20;
        ++v12;
      }
      while ( v14 != v13 );
      v7 = v31;
      v9 = (unsigned int)v32;
    }
    else
    {
      v7 = v31;
    }
  }
  v23 = *(_QWORD *)(a2 + 80);
  v24 = *(_QWORD **)(a1 + 8);
  v29 = v23;
  if ( v23 )
  {
    v28 = v7;
    sub_B96E90((__int64)&v29, v23, 1);
    v7 = v28;
  }
  v30 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v27 + 1) = v9;
  *(_QWORD *)&v27 = v7;
  v25 = sub_33FC220(
          v24,
          156,
          (__int64)&v29,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          (__int64)v7,
          v27);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
  return v25;
}
