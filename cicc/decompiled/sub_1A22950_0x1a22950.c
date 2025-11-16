// Function: sub_1A22950
// Address: 0x1a22950
//
void __fastcall sub_1A22950(__int64 *a1, __int64 a2)
{
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  _BYTE *v6; // rdi
  unsigned int v7; // eax
  unsigned int v8; // edx
  __int64 v9; // rsi
  unsigned int v10; // eax
  __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // r15
  char v15; // dl
  unsigned int v16; // edx
  __int64 v17; // rsi
  int v18; // eax
  __int64 v19; // rcx
  unsigned int v20; // esi
  __int64 v21; // r15
  int v22; // r8d
  int v23; // r9d
  char v24; // dl
  _QWORD *v25; // r14
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  _QWORD *v28; // rcx
  __int64 v29; // rax
  unsigned int v30; // [rsp+1Ch] [rbp-E4h]
  unsigned int v31; // [rsp+1Ch] [rbp-E4h]
  char v32[48]; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE *v33; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+58h] [rbp-A8h]
  _BYTE v35[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v37; // [rsp+88h] [rbp-78h]
  _BYTE *v38; // [rsp+90h] [rbp-70h]
  __int64 v39; // [rsp+98h] [rbp-68h]
  int v40; // [rsp+A0h] [rbp-60h]
  _BYTE v41[88]; // [rsp+A8h] [rbp-58h] BYREF

  v37 = v41;
  v38 = v41;
  v33 = v35;
  v34 = 0x400000000LL;
  v36 = 0;
  v39 = 4;
  v40 = 0;
  sub_165A590((__int64)v32, (__int64)&v36, a2);
  v5 = (unsigned int)v34;
  if ( (unsigned int)v34 >= HIDWORD(v34) )
  {
    sub_16CD150((__int64)&v33, v35, 0, 8, v3, v4);
    v5 = (unsigned int)v34;
  }
  *(_QWORD *)&v33[8 * v5] = a2;
  v6 = v33;
  v7 = v34 + 1;
  LODWORD(v34) = v34 + 1;
  do
  {
    while ( 1 )
    {
      v13 = v7--;
      v14 = *(_QWORD *)&v6[8 * v13 - 8];
      LODWORD(v34) = v7;
      v15 = *(_BYTE *)(v14 + 16);
      if ( v15 == 54 )
      {
        v8 = 1 << (*(unsigned __int16 *)(v14 + 18) >> 1) >> 1;
        if ( !v8 )
          v8 = sub_15A9FE0(*a1, *(_QWORD *)v14);
        v9 = a1[6];
        v10 = (unsigned int)(1 << *(_WORD *)(v9 + 18)) >> 1;
        if ( !v10 )
        {
          v30 = v8;
          v10 = sub_15A9FE0(*a1, *(_QWORD *)(v9 + 56));
          v8 = v30;
        }
        v11 = (a1[16] - a1[7]) | v10;
        v12 = -(int)v11 & v11;
        if ( v8 < v12 )
          v12 = v8;
        sub_15F8F50(v14, v12);
        v7 = v34;
        v6 = v33;
        goto LABEL_11;
      }
      if ( v15 == 55 )
        break;
      v21 = *(_QWORD *)(v14 + 8);
      if ( v21 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v25 = sub_1648700(v21);
            v26 = v37;
            if ( v38 != v37 )
              break;
            v27 = &v37[8 * HIDWORD(v39)];
            if ( v37 != (_BYTE *)v27 )
            {
              v28 = 0;
              while ( v25 != (_QWORD *)*v26 )
              {
                if ( *v26 == -2 )
                  v28 = v26;
                if ( v27 == ++v26 )
                {
                  if ( !v28 )
                    goto LABEL_42;
                  *v28 = v25;
                  --v40;
                  ++v36;
                  goto LABEL_39;
                }
              }
              goto LABEL_29;
            }
LABEL_42:
            if ( HIDWORD(v39) >= (unsigned int)v39 )
              break;
            ++HIDWORD(v39);
            *v27 = v25;
            v29 = (unsigned int)v34;
            ++v36;
            if ( (unsigned int)v34 >= HIDWORD(v34) )
            {
LABEL_44:
              sub_16CD150((__int64)&v33, v35, 0, 8, v22, v23);
              v29 = (unsigned int)v34;
            }
LABEL_40:
            *(_QWORD *)&v33[8 * v29] = v25;
            LODWORD(v34) = v34 + 1;
            v21 = *(_QWORD *)(v21 + 8);
            if ( !v21 )
            {
LABEL_41:
              v7 = v34;
              v6 = v33;
              goto LABEL_11;
            }
          }
          sub_16CCBA0((__int64)&v36, (__int64)v25);
          if ( v24 )
          {
LABEL_39:
            v29 = (unsigned int)v34;
            if ( (unsigned int)v34 >= HIDWORD(v34) )
              goto LABEL_44;
            goto LABEL_40;
          }
LABEL_29:
          v21 = *(_QWORD *)(v21 + 8);
          if ( !v21 )
            goto LABEL_41;
        }
      }
LABEL_11:
      if ( !v7 )
        goto LABEL_21;
    }
    v16 = 1 << (*(unsigned __int16 *)(v14 + 18) >> 1) >> 1;
    if ( !v16 )
      v16 = sub_15A9FE0(*a1, **(_QWORD **)(v14 - 48));
    v17 = a1[6];
    v18 = (unsigned int)(1 << *(_WORD *)(v17 + 18)) >> 1;
    if ( !v18 )
    {
      v31 = v16;
      v18 = sub_15A9FE0(*a1, *(_QWORD *)(v17 + 56));
      v16 = v31;
    }
    v19 = a1[16] - a1[7];
    v20 = -(v19 | v18) & (v19 | v18);
    if ( v16 < v20 )
      v20 = v16;
    sub_15F9450(v14, v20);
    v7 = v34;
    v6 = v33;
  }
  while ( (_DWORD)v34 );
LABEL_21:
  if ( v6 != v35 )
    _libc_free((unsigned __int64)v6);
  if ( v38 != v37 )
    _libc_free((unsigned __int64)v38);
}
