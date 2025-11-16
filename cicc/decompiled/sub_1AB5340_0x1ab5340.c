// Function: sub_1AB5340
// Address: 0x1ab5340
//
__int64 __fastcall sub_1AB5340(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r13
  __int64 v8; // r9
  __int64 v9; // rax
  char v10; // di
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r14
  _QWORD *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 v22; // rdx
  _QWORD *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // r9
  unsigned int v32; // r8d
  __int64 v33; // rdi
  __int64 v34; // r11
  __int64 v35; // rcx
  __int64 v36; // rdi
  unsigned __int64 v37; // rsi
  __int64 v38; // rsi
  int v39; // edi
  int v41; // [rsp+4h] [rbp-8Ch]
  __int64 v42; // [rsp+8h] [rbp-88h]
  unsigned __int64 v44; // [rsp+18h] [rbp-78h]
  const char *v47; // [rsp+30h] [rbp-60h] BYREF
  __int64 v48; // [rsp+38h] [rbp-58h]
  __int64 v49[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v50; // [rsp+50h] [rbp-40h]

  for ( i = a1[6]; ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v8 = i - 24;
    v9 = 0x17FFFFFFE8LL;
    v10 = *(_BYTE *)(i - 1) & 0x40;
    v11 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v11 )
    {
      v12 = 24LL * *(unsigned int *)(i + 32) + 8;
      v13 = 0;
      do
      {
        v14 = v8 - 24LL * v11;
        if ( v10 )
          v14 = *(_QWORD *)(i - 32);
        if ( a2 == *(_QWORD **)(v14 + v12) )
        {
          v9 = 24 * v13;
          goto LABEL_11;
        }
        ++v13;
        v12 += 8;
      }
      while ( v11 != (_DWORD)v13 );
      v9 = 0x17FFFFFFE8LL;
    }
LABEL_11:
    if ( v10 )
      v15 = *(_QWORD *)(i - 32);
    else
      v15 = v8 - 24LL * v11;
    v16 = *(_QWORD *)(v15 + v9);
    v17 = sub_1AB4240(a4, i - 24);
    v18 = v17[2];
    if ( v18 != v16 )
    {
      if ( v18 != -8 && v18 != 0 && v18 != -16 )
        sub_1649B30(v17);
      v17[2] = v16;
      if ( v16 != 0 && v16 != -8 && v16 != -16 )
        sub_164C220((__int64)v17);
    }
  }
  v42 = sub_1AA91E0(a2, a1, a5, 0);
  v47 = sub_1649960((__int64)a2);
  v48 = v19;
  v49[0] = (__int64)&v47;
  v50 = 773;
  v49[1] = (__int64)".split";
  sub_164B780(v42, v49);
  v44 = sub_157EBA0(v42);
  while ( 1 )
  {
    v20 = i - 24;
    if ( !i )
      v20 = 0;
    if ( v20 == a3 || v20 == sub_157EBA0((__int64)a1) )
      return v42;
    v21 = sub_15F4880(v20);
    v47 = sub_1649960(v20);
    v50 = 261;
    v48 = v22;
    v49[0] = (__int64)&v47;
    sub_164B780(v21, v49);
    sub_15F2120(v21, v44);
    v23 = sub_1AB4240(a4, v20);
    v24 = v23[2];
    if ( v21 != v24 )
    {
      if ( v24 != 0 && v24 != -8 && v24 != -16 )
        sub_1649B30(v23);
      v23[2] = v21;
      if ( v21 != 0 && v21 != -8 && v21 != -16 )
        sub_164C220((__int64)v23);
    }
    if ( (*(_DWORD *)(v21 + 20) & 0xFFFFFFF) != 0 )
    {
      v25 = 0;
      v26 = 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
          v27 = *(_QWORD *)(v21 - 8);
        else
          v27 = v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
        v28 = (_QWORD *)(v25 + v27);
        v29 = *v28;
        if ( *(_BYTE *)(*v28 + 16LL) > 0x17u )
        {
          v30 = *(unsigned int *)(a4 + 24);
          if ( (_DWORD)v30 )
          {
            v31 = *(_QWORD *)(a4 + 8);
            v32 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v33 = v31 + ((unsigned __int64)v32 << 6);
            v34 = *(_QWORD *)(v33 + 24);
            if ( v29 == v34 )
            {
LABEL_39:
              if ( v33 != v31 + (v30 << 6) )
              {
                v35 = *(_QWORD *)(v33 + 56);
                v36 = v28[1];
                v37 = v28[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v37 = v36;
                if ( v36 )
                  *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
                *v28 = v35;
                if ( v35 )
                {
                  v38 = *(_QWORD *)(v35 + 8);
                  v28[1] = v38;
                  if ( v38 )
                    *(_QWORD *)(v38 + 16) = (unsigned __int64)(v28 + 1) | *(_QWORD *)(v38 + 16) & 3LL;
                  v28[2] = (v35 + 8) | v28[2] & 3LL;
                  *(_QWORD *)(v35 + 8) = v28;
                }
              }
            }
            else
            {
              v39 = 1;
              while ( v34 != -8 )
              {
                v32 = (v30 - 1) & (v39 + v32);
                v41 = v39 + 1;
                v33 = v31 + ((unsigned __int64)v32 << 6);
                v34 = *(_QWORD *)(v33 + 24);
                if ( v29 == v34 )
                  goto LABEL_39;
                v39 = v41;
              }
            }
          }
        }
        v25 += 24;
      }
      while ( v25 != v26 );
    }
    i = *(_QWORD *)(i + 8);
  }
}
