// Function: sub_10584A0
// Address: 0x10584a0
//
__int64 __fastcall sub_10584A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rsi
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 result; // rax
  __int64 v18; // r14
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 *v22; // r14
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r12
  __int64 i; // r13
  __int64 *v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // [rsp+8h] [rbp-88h]
  __int64 *v31; // [rsp+10h] [rbp-80h]
  __int64 *v32; // [rsp+18h] [rbp-78h]
  __int64 *v33; // [rsp+18h] [rbp-78h]
  __int64 *v34; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h]
  _BYTE v36[96]; // [rsp+30h] [rbp-60h] BYREF

  v7 = &v34;
  v34 = (__int64 *)v36;
  v35 = 0x600000000LL;
  sub_E388C0(a2, (__int64)&v34, a3, a4, a5, a6);
  v8 = &v34[(unsigned int)v35];
  if ( v34 != v8 )
  {
    v32 = v34;
    do
    {
      v9 = sub_AA5930(*v32);
      v11 = v10;
      v12 = v9;
LABEL_4:
      if ( v11 != v12 )
      {
        while ( 1 )
        {
          v7 = (_QWORD *)v12;
          if ( !sub_1056220(a1, v12, a2) )
            goto LABEL_6;
          if ( !*(_BYTE *)(a1 + 1284) )
            break;
          v28 = *(__int64 **)(a1 + 1264);
          v29 = &v28[*(unsigned int *)(a1 + 1276)];
          if ( v28 == v29 )
            goto LABEL_36;
          while ( v12 != *v28 )
          {
            if ( v29 == ++v28 )
              goto LABEL_36;
          }
LABEL_6:
          if ( !v12 )
            BUG();
          v16 = *(_QWORD *)(v12 + 32);
          if ( !v16 )
            BUG();
          v12 = 0;
          if ( *(_BYTE *)(v16 - 24) != 84 )
            goto LABEL_4;
          v12 = v16 - 24;
          if ( v11 == v16 - 24 )
            goto LABEL_10;
        }
        v7 = (_QWORD *)v12;
        if ( sub_C8CA60(a1 + 1256, v12) )
          goto LABEL_6;
LABEL_36:
        v7 = (_QWORD *)v12;
        sub_1057F60(a1, (unsigned __int8 *)v12, v29, v13, v14, v15);
        goto LABEL_6;
      }
LABEL_10:
      ++v32;
    }
    while ( v8 != v32 );
    v8 = v34;
  }
  result = *(_QWORD *)(a2 + 88);
  v30 = (__int64 *)(result + 8LL * *(unsigned int *)(a2 + 96));
  if ( (__int64 *)result != v30 )
  {
    v33 = *(__int64 **)(a2 + 88);
    do
    {
      v18 = 8LL * (unsigned int)v35;
      v19 = *v33;
      v31 = &v8[(unsigned __int64)v18 / 8];
      v20 = v18 >> 3;
      v21 = v18 >> 5;
      if ( v21 )
      {
        v22 = &v8[4 * v21];
        while ( 1 )
        {
          v7 = (_QWORD *)v19;
          if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, *v8) )
            break;
          v7 = (_QWORD *)v19;
          if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, v8[1]) )
          {
            ++v8;
            break;
          }
          v7 = (_QWORD *)v19;
          if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, v8[2]) )
          {
            v8 += 2;
            break;
          }
          v7 = (_QWORD *)v19;
          if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, v8[3]) )
          {
            v8 += 3;
            break;
          }
          v8 += 4;
          if ( v22 == v8 )
          {
            v20 = v31 - v8;
            goto LABEL_38;
          }
        }
LABEL_21:
        if ( v31 != v8 )
        {
          v26 = *(_QWORD *)(v19 + 56);
          for ( i = v19 + 48; i != v26; v26 = *(_QWORD *)(v26 + 8) )
          {
            v7 = (_QWORD *)(v26 - 24);
            if ( !v26 )
              v7 = 0;
            sub_1058220(a1, (__int64)v7, a2, v23, v24, v25);
          }
        }
        goto LABEL_26;
      }
LABEL_38:
      if ( v20 != 2 )
      {
        if ( v20 != 3 )
        {
          if ( v20 != 1 )
            goto LABEL_26;
          goto LABEL_41;
        }
        v7 = (_QWORD *)v19;
        if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, *v8) )
          goto LABEL_21;
        ++v8;
      }
      v7 = (_QWORD *)v19;
      if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, *v8) )
        goto LABEL_21;
      ++v8;
LABEL_41:
      v7 = (_QWORD *)v19;
      if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 584), v19, *v8) )
        goto LABEL_21;
LABEL_26:
      ++v33;
      v8 = v34;
      result = (__int64)v33;
    }
    while ( v30 != v33 );
  }
  if ( v8 != (__int64 *)v36 )
    return _libc_free(v8, v7);
  return result;
}
