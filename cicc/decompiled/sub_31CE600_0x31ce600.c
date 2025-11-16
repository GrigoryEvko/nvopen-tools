// Function: sub_31CE600
// Address: 0x31ce600
//
__int64 __fastcall sub_31CE600(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 *v9; // rsi
  __int64 *v10; // r14
  __int64 v11; // r13
  unsigned __int8 *v12; // rsi
  int v13; // edx
  _BYTE *v14; // rcx
  __int64 v15; // rdi
  __int64 *v17; // [rsp+0h] [rbp-1A0h]
  __int64 *v18; // [rsp+10h] [rbp-190h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-188h]
  __int64 *v20; // [rsp+20h] [rbp-180h]
  _BYTE *v21; // [rsp+30h] [rbp-170h] BYREF
  __int64 v22; // [rsp+38h] [rbp-168h]
  _BYTE v23[256]; // [rsp+40h] [rbp-160h] BYREF
  __int64 v24; // [rsp+140h] [rbp-60h]
  __int64 v25; // [rsp+148h] [rbp-58h]
  __int64 v26; // [rsp+150h] [rbp-50h]
  __int64 v27; // [rsp+158h] [rbp-48h]
  __int64 v28; // [rsp+160h] [rbp-40h]
  __int64 v29; // [rsp+168h] [rbp-38h]

  v2 = a2 + 72;
  v3 = *(_QWORD *)(a2 + 80);
  v18 = 0;
  v19 = 0;
  v20 = 0;
  if ( v3 == a2 + 72 )
    return 1;
  do
  {
    if ( !v3 )
LABEL_43:
      BUG();
    v4 = *(_QWORD *)(v3 + 32);
    v5 = v3 + 24;
    if ( v4 != v3 + 24 )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        if ( *(_BYTE *)(v4 - 24) != 85 )
          goto LABEL_5;
        v6 = *(_QWORD *)(v4 - 56);
        if ( !v6 )
          goto LABEL_5;
        if ( *(_BYTE *)v6 )
          goto LABEL_5;
        v7 = *(_QWORD *)(v4 + 56);
        if ( *(_QWORD *)(v6 + 24) != v7 || (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
          goto LABEL_5;
        v21 = (_BYTE *)(v4 - 24);
        v8 = *(_QWORD *)(v4 - 56);
        if ( !v8 || *(_BYTE *)v8 || v7 != *(_QWORD *)(v8 + 24) )
          BUG();
        if ( *(_DWORD *)(v8 + 36) != 8170 )
          goto LABEL_5;
        v9 = v19;
        if ( v19 == v20 )
        {
          sub_2CB4B10((__int64)&v18, v19, &v21);
LABEL_5:
          v4 = *(_QWORD *)(v4 + 8);
          if ( v5 == v4 )
            break;
        }
        else
        {
          if ( v19 )
          {
            *v19 = v4 - 24;
            v9 = v19;
          }
          v19 = v9 + 1;
          v4 = *(_QWORD *)(v4 + 8);
          if ( v5 == v4 )
            break;
        }
      }
    }
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v2 != v3 );
  v10 = v18;
  v17 = v19;
  if ( v19 != v18 )
  {
    do
    {
      v11 = *v10;
      v21 = v23;
      v22 = 0x2000000000LL;
      v24 = 0;
      v12 = (unsigned __int8 *)v11;
      v25 = 0;
      v26 = 0;
      v27 = 0;
      v28 = v11;
      v29 = 0;
LABEL_23:
      sub_31CE310((__int64)&v21, (__int64)v12);
LABEL_24:
      v13 = v22;
LABEL_25:
      v14 = &v21[8 * v13];
      while ( v13 )
      {
        v15 = *((_QWORD *)v14 - 1);
        --v13;
        v14 -= 8;
        LODWORD(v22) = v13;
        v29 = v15;
        v12 = *(unsigned __int8 **)(v15 + 24);
        if ( *v12 > 0x1Cu )
        {
          switch ( *v12 )
          {
            case 0x1Eu:
            case 0x1Fu:
            case 0x20u:
            case 0x21u:
            case 0x22u:
            case 0x23u:
            case 0x24u:
            case 0x25u:
            case 0x26u:
            case 0x27u:
            case 0x28u:
            case 0x29u:
            case 0x2Au:
            case 0x2Bu:
            case 0x2Cu:
            case 0x2Du:
            case 0x2Eu:
            case 0x2Fu:
            case 0x30u:
            case 0x31u:
            case 0x32u:
            case 0x33u:
            case 0x34u:
            case 0x35u:
            case 0x36u:
            case 0x37u:
            case 0x38u:
            case 0x39u:
            case 0x3Au:
            case 0x3Bu:
            case 0x3Cu:
            case 0x40u:
            case 0x41u:
            case 0x43u:
            case 0x44u:
            case 0x45u:
            case 0x46u:
            case 0x47u:
            case 0x48u:
            case 0x49u:
            case 0x4Au:
            case 0x4Bu:
            case 0x4Cu:
            case 0x4Du:
            case 0x50u:
            case 0x51u:
            case 0x52u:
            case 0x53u:
            case 0x57u:
            case 0x58u:
            case 0x59u:
            case 0x5Au:
            case 0x5Bu:
            case 0x5Cu:
            case 0x5Du:
            case 0x5Eu:
            case 0x5Fu:
            case 0x60u:
              goto LABEL_25;
            case 0x3Du:
              sub_31CD360((__int64)&v21, (__int64)v12);
              goto LABEL_24;
            case 0x3Eu:
              sub_31CD500((__int64)&v21, v12);
              goto LABEL_24;
            case 0x3Fu:
              v12 = *(unsigned __int8 **)(v15 + 24);
              if ( !(unsigned int)sub_BD2910(v15) )
                goto LABEL_23;
              goto LABEL_24;
            case 0x42u:
              sub_31CDA70((__int64)&v21, (__int64)v12);
              goto LABEL_24;
            case 0x4Eu:
            case 0x4Fu:
            case 0x54u:
            case 0x56u:
              goto LABEL_23;
            case 0x55u:
              sub_31CE2C0((__int64)&v21, (__int64)v12);
              goto LABEL_24;
            default:
              goto LABEL_43;
          }
        }
      }
      sub_C7D6A0(v25, 8LL * (unsigned int)v27, 8);
      if ( v21 != v23 )
        _libc_free((unsigned __int64)v21);
      ++v10;
      sub_BD84D0(v11, *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
      sub_B43D60((_QWORD *)v11);
    }
    while ( v17 != v10 );
    v10 = v18;
  }
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
  return 1;
}
