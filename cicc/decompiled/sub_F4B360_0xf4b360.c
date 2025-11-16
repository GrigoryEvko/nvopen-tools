// Function: sub_F4B360
// Address: 0xf4b360
//
__int64 __fastcall sub_F4B360(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, _BYTE *a5)
{
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rbx
  char v11; // si
  unsigned __int8 *v12; // r15
  _QWORD *v13; // rdi
  unsigned __int8 *v14; // rax
  char v15; // al
  const char *v16; // rax
  __int64 v17; // rdx
  char v18; // cl
  __int64 *v19; // rsi
  const char *v21; // rax
  __int64 v22; // rdx
  char v23; // cl
  __int64 *v24; // rsi
  __int64 v26; // [rsp+10h] [rbp-A0h]
  char v27; // [rsp+1Eh] [rbp-92h]
  char v28; // [rsp+1Fh] [rbp-91h]
  __int64 v31; // [rsp+40h] [rbp-70h]
  char v33; // [rsp+48h] [rbp-68h]
  const char *v34; // [rsp+50h] [rbp-60h] BYREF
  __int64 v35; // [rsp+58h] [rbp-58h]
  __int64 *v36; // [rsp+60h] [rbp-50h]
  __int64 v37; // [rsp+68h] [rbp-48h]
  __int16 v38; // [rsp+70h] [rbp-40h]

  v38 = 257;
  v7 = sub_AA48A0(a1);
  v8 = sub_22077B0(80);
  v9 = v8;
  if ( v8 )
    sub_AA4D50(v8, v7, (__int64)&v34, a4, 0);
  *(_BYTE *)(v9 + 40) = *(_BYTE *)(a1 + 40);
  if ( (*(_BYTE *)(a1 + 7) & 0x10) != 0 )
  {
    v21 = sub_BD5D20(a1);
    v23 = *((_BYTE *)a3 + 32);
    if ( v23 )
    {
      if ( v23 == 1 )
      {
        v34 = v21;
        v35 = v22;
        v38 = 261;
      }
      else
      {
        if ( *((_BYTE *)a3 + 33) == 1 )
        {
          v6 = a3[1];
          v24 = (__int64 *)*a3;
        }
        else
        {
          v24 = a3;
          v23 = 2;
        }
        v34 = v21;
        v35 = v22;
        v36 = v24;
        v37 = v6;
        LOBYTE(v38) = 5;
        HIBYTE(v38) = v23;
      }
    }
    else
    {
      v38 = 256;
    }
    sub_BD6B50((unsigned __int8 *)v9, &v34);
  }
  v10 = *(_QWORD *)(a1 + 56);
  v33 = 0;
  v31 = a1 + 48;
  v27 = 0;
  v28 = 0;
  if ( a1 + 48 != v10 )
  {
    while ( 1 )
    {
      if ( !v10 )
      {
        sub_B47F80(0);
        BUG();
      }
      v12 = (unsigned __int8 *)sub_B47F80((_BYTE *)(v10 - 24));
      if ( (*(_BYTE *)(v10 - 17) & 0x10) != 0 )
      {
        v16 = sub_BD5D20(v10 - 24);
        v18 = *((_BYTE *)a3 + 32);
        if ( v18 )
        {
          if ( v18 == 1 )
          {
            v34 = v16;
            v35 = v17;
            v38 = 261;
          }
          else
          {
            if ( *((_BYTE *)a3 + 33) == 1 )
            {
              v19 = (__int64 *)*a3;
              v26 = a3[1];
            }
            else
            {
              v19 = a3;
              v18 = 2;
            }
            v34 = v16;
            v35 = v17;
            v36 = v19;
            v37 = v26;
            LOBYTE(v38) = 5;
            HIBYTE(v38) = v18;
          }
        }
        else
        {
          v38 = 256;
        }
        sub_BD6B50(v12, &v34);
      }
      LOWORD(v5) = 0;
      sub_B44150(v12, v9, (unsigned __int64 *)(v9 + 48), v5);
      LOBYTE(v35) = 0;
      sub_B43F50((__int64)v12, v10 - 24, (__int64)v34, 0, 0);
      v13 = sub_F46C80(a2, v10 - 24);
      v14 = (unsigned __int8 *)v13[2];
      if ( v12 != v14 )
      {
        if ( v14 + 4096 != 0 && v14 != 0 && v14 != (unsigned __int8 *)-8192LL )
          sub_BD60C0(v13);
        v13[2] = v12;
        if ( v12 + 4096 != 0 && v12 != 0 && v12 != (unsigned __int8 *)-8192LL )
          sub_BD73F0((__int64)v13);
      }
      v15 = *(_BYTE *)(v10 - 24);
      if ( v15 != 85 )
        goto LABEL_6;
      if ( !sub_B46AA0(v10 - 24) )
      {
        v28 = 1;
        if ( (*(_BYTE *)(v10 - 17) & 0x20) != 0 )
          break;
      }
LABEL_10:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v31 == v10 )
        goto LABEL_33;
    }
    v27 |= sub_B91C10(v10 - 24, 34) != 0;
    if ( (*(_BYTE *)(v10 - 17) & 0x20) != 0 )
      v27 |= sub_B91C10(v10 - 24, 35) != 0;
    v28 = 1;
    v15 = *(_BYTE *)(v10 - 24);
LABEL_6:
    if ( v15 == 60 )
    {
      v11 = v33;
      if ( !sub_B4D040(v10 - 24) )
        v11 = 1;
      v33 = v11;
    }
    goto LABEL_10;
  }
LABEL_33:
  if ( a5 )
  {
    *a5 |= v28;
    a5[2] |= v33;
    a5[1] |= v27;
  }
  return v9;
}
