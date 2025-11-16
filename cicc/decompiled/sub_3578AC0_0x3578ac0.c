// Function: sub_3578AC0
// Address: 0x3578ac0
//
void __fastcall sub_3578AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rax
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 *v20; // r14
  __int64 v21; // r12
  __int64 v22; // r13
  __int64 *v23; // [rsp+8h] [rbp-88h]
  __int64 *v24; // [rsp+10h] [rbp-80h]
  __int64 *v25; // [rsp+18h] [rbp-78h]
  __int64 *v26; // [rsp+18h] [rbp-78h]
  __int64 *v27; // [rsp+20h] [rbp-70h] BYREF
  __int64 v28; // [rsp+28h] [rbp-68h]
  _BYTE v29[96]; // [rsp+30h] [rbp-60h] BYREF

  v27 = (__int64 *)v29;
  v28 = 0x600000000LL;
  sub_2E5E840(a2, (__int64)&v27, a3, a4, a5, a6);
  v6 = &v27[(unsigned int)v28];
  if ( v27 != v6 )
  {
    v25 = v27;
    while ( 1 )
    {
      v7 = *v25;
      v8 = sub_2E311E0(*v25);
      v9 = *(_QWORD *)(v7 + 56);
      v10 = v8;
      if ( v9 != v8 )
        break;
LABEL_11:
      if ( v6 == ++v25 )
      {
        v6 = v27;
        goto LABEL_13;
      }
    }
    while ( 1 )
    {
      if ( (unsigned __int8)sub_3574DA0(a1, v9, a2) )
        sub_3577FF0(a1, v9, v11, v12, v13, v14);
      if ( !v9 )
        break;
      if ( (*(_BYTE *)v9 & 4) != 0 )
      {
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          goto LABEL_11;
      }
      else
      {
        while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
          v9 = *(_QWORD *)(v9 + 8);
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          goto LABEL_11;
      }
    }
LABEL_52:
    BUG();
  }
LABEL_13:
  v15 = *(__int64 **)(a2 + 88);
  v23 = &v15[*(unsigned int *)(a2 + 96)];
  if ( v15 != v23 )
  {
    v26 = *(__int64 **)(a2 + 88);
    while ( 1 )
    {
      v16 = 8LL * (unsigned int)v28;
      v17 = *v26;
      v24 = &v6[(unsigned __int64)v16 / 8];
      v18 = v16 >> 3;
      v19 = v16 >> 5;
      if ( !v19 )
        goto LABEL_40;
      v20 = &v6[4 * v19];
      do
      {
        if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, *v6) )
          goto LABEL_22;
        if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, v6[1]) )
        {
          ++v6;
          goto LABEL_22;
        }
        if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, v6[2]) )
        {
          v6 += 2;
          goto LABEL_22;
        }
        if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, v6[3]) )
        {
          v6 += 3;
          goto LABEL_22;
        }
        v6 += 4;
      }
      while ( v20 != v6 );
      v18 = v24 - v6;
LABEL_40:
      if ( v18 == 2 )
        goto LABEL_50;
      if ( v18 != 3 )
      {
        if ( v18 == 1 )
          goto LABEL_43;
        goto LABEL_29;
      }
      if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, *v6) )
        break;
      ++v6;
LABEL_50:
      if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, *v6) )
        break;
      ++v6;
LABEL_43:
      if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 584), v17, *v6) )
        break;
LABEL_29:
      ++v26;
      v6 = v27;
      if ( v23 == v26 )
        goto LABEL_30;
    }
LABEL_22:
    if ( v24 == v6 )
      goto LABEL_29;
    v21 = *(_QWORD *)(v17 + 56);
    v22 = v17 + 48;
    if ( v22 == v21 )
      goto LABEL_29;
    while ( 1 )
    {
      sub_3578760(a1, v21, a2);
      if ( !v21 )
        goto LABEL_52;
      if ( (*(_BYTE *)v21 & 4) != 0 )
      {
        v21 = *(_QWORD *)(v21 + 8);
        if ( v22 == v21 )
          goto LABEL_29;
      }
      else
      {
        while ( (*(_BYTE *)(v21 + 44) & 8) != 0 )
          v21 = *(_QWORD *)(v21 + 8);
        v21 = *(_QWORD *)(v21 + 8);
        if ( v22 == v21 )
          goto LABEL_29;
      }
    }
  }
LABEL_30:
  if ( v6 != (__int64 *)v29 )
    _libc_free((unsigned __int64)v6);
}
