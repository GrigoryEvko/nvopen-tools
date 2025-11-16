// Function: sub_1422A60
// Address: 0x1422a60
//
void __fastcall sub_1422A60(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // edx
  int v26; // r9d
  int v27; // edx
  int v28; // r9d
  __int64 v29; // [rsp+18h] [rbp-268h]
  __int64 v30; // [rsp+20h] [rbp-260h]
  __int64 v31; // [rsp+20h] [rbp-260h]
  __int64 v32; // [rsp+20h] [rbp-260h]
  __int64 v33; // [rsp+28h] [rbp-258h]
  _BYTE *v34; // [rsp+30h] [rbp-250h] BYREF
  __int64 v35; // [rsp+38h] [rbp-248h]
  _BYTE v36[256]; // [rsp+40h] [rbp-240h] BYREF
  _BYTE *v37; // [rsp+140h] [rbp-140h] BYREF
  __int64 v38; // [rsp+148h] [rbp-138h]
  _BYTE v39[304]; // [rsp+150h] [rbp-130h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v34 = v36;
  v35 = 0x2000000000LL;
  v38 = 0x2000000000LL;
  v37 = v39;
  v29 = a2 + 72;
  if ( a2 + 72 == v2 )
    return;
  do
  {
    v3 = *(unsigned int *)(a1 + 80);
    v4 = v2 - 24;
    v33 = 0;
    if ( !v2 )
      v4 = 0;
    if ( (_DWORD)v3 )
    {
      v5 = *(_QWORD *)(a1 + 64);
      v6 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( v4 == *v7 )
      {
LABEL_6:
        if ( v7 != (__int64 *)(v5 + 16 * v3) )
        {
          v33 = v7[1];
          goto LABEL_8;
        }
      }
      else
      {
        v27 = 1;
        while ( v8 != -8 )
        {
          v28 = v27 + 1;
          v6 = (v3 - 1) & (v27 + v6);
          v7 = (__int64 *)(v5 + 16LL * v6);
          v8 = *v7;
          if ( v4 == *v7 )
            goto LABEL_6;
          v27 = v28;
        }
      }
      v33 = 0;
    }
LABEL_8:
    v9 = *(unsigned int *)(a1 + 112);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD *)(a1 + 96);
      v11 = (v9 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v4 == *v12 )
      {
LABEL_10:
        if ( v12 != (__int64 *)(v10 + 16 * v9) )
          v33 |= v12[1];
      }
      else
      {
        v25 = 1;
        while ( v13 != -8 )
        {
          v26 = v25 + 1;
          v11 = (v9 - 1) & (v25 + v11);
          v12 = (__int64 *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v4 == *v12 )
            goto LABEL_10;
          v25 = v26;
        }
      }
    }
    v14 = sub_14228C0(a1, v4);
    if ( v14 )
    {
      v15 = (unsigned int)v35;
      if ( (unsigned int)v35 >= HIDWORD(v35) )
      {
        v32 = v14;
        sub_16CD150(&v34, v36, 0, 8);
        v15 = (unsigned int)v35;
        v14 = v32;
      }
      *(_QWORD *)&v34[8 * v15] = v14;
      v16 = (unsigned int)v38;
      LODWORD(v35) = v35 + 1;
      if ( (unsigned int)v38 >= HIDWORD(v38) )
      {
        v31 = v14;
        sub_16CD150(&v37, v39, 0, 8);
        v16 = (unsigned int)v38;
        v14 = v31;
      }
      *(_QWORD *)&v37[8 * v16] = v14;
      LODWORD(v38) = v38 + 1;
    }
    v17 = *(_QWORD *)(v4 + 48);
    v18 = v4 + 40;
    if ( v18 != v17 )
    {
      v30 = v2;
      v19 = v17;
      v20 = v18;
      do
      {
        v21 = v19 - 24;
        if ( !v19 )
          v21 = 0;
        v22 = sub_1422850(a1, v21);
        if ( v22 )
        {
          v23 = (unsigned int)v35;
          if ( (unsigned int)v35 >= HIDWORD(v35) )
          {
            sub_16CD150(&v34, v36, 0, 8);
            v23 = (unsigned int)v35;
          }
          *(_QWORD *)&v34[8 * v23] = v22;
          LODWORD(v35) = v35 + 1;
          if ( *(_BYTE *)(v22 + 16) == 22 )
          {
            v24 = (unsigned int)v38;
            if ( (unsigned int)v38 >= HIDWORD(v38) )
            {
              sub_16CD150(&v37, v39, 0, 8);
              v24 = (unsigned int)v38;
            }
            *(_QWORD *)&v37[8 * v24] = v22;
            LODWORD(v38) = v38 + 1;
          }
        }
        v19 = *(_QWORD *)(v19 + 8);
      }
      while ( v20 != v19 );
      v2 = v30;
    }
    if ( v33 )
    {
      LODWORD(v35) = 0;
      LODWORD(v38) = 0;
    }
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v29 != v2 );
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
